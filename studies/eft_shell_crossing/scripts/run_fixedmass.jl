# Fixed-mass sheet statistics of LPT and PM runs (src/fixedmass.jl); one HDF5 file per job.
#
#   julia --project=. scripts/run_fixedmass.jl config/fm_<name>.toml [lpt|pm|all] [max_jobs]
#
# The config names a `base` configuration (ICs, cosmology, solvers, epochs) and adds a [fixedmass]
# table; [seeds] there overrides the base seeds.  Output: $EFTSC_SCRATCH/<name>/fm_<model>_<member>_
# L<seedL>_S<seedS>_s<±1>.h5 with one group per snapshot ("e<epoch index>_o<LPT order | 0 for PM>").
# Restartable: existing files are skipped.
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, CUDA, DiscoDJNative, HDF5, TOML, Printf, Dates
const DJ = DiscoDJNative

scratch() = get(ENV, "EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")

function load_cfg(path)
    fm = TOML.parsefile(path)
    cfg = TOML.parsefile(joinpath(dirname(path), fm["base"]))
    cfg["name"] = fm["name"]; cfg["fixedmass"] = fm["fixedmass"]
    haskey(fm, "seeds") && (cfg["seeds"] = merge(cfg["seeds"], fm["seeds"]); delete!(cfg["seeds"], "extra"))
    cfg["config_fm"] = sprint(TOML.print, fm)
    return cfg
end

function jobs(cfg, model)
    sd = cfg["seeds"]; out = []
    for m in (model == "all" ? ("lpt", "pm") : (model,)), sL in sd["long"], sg in sd["signs"]
        push!(out, (m, :IR, sL, 0, sg))
        for sS in sd["short"]; push!(out, (m, :UV, sL, sS, sg)); end
    end
    return out
end
fname(cfg, j) = joinpath(scratch(), cfg["name"], @sprintf("fm_%s_%s_L%d_S%d_s%+d.h5", j[1], j[2], j[3], j[4], j[5]))
spec_of(cfg, N, sL, sS, sg) = PairSpec(N=N, N_ic=cfg["N_ic"], L=cfg["L"], k_split=cfg["k_split"], k_uv=cfg["k_uv"],
                                       long_sign=Int(sign(sg)), seed_L=sL, seed_S=sS == 0 ? 1 : sS)

function fm_plan(cfg, dev)
    f = cfg["fixedmass"]
    R = exp.(range(log(f["R_min"]), log(f["R_max"]), length=f["n_R"]))
    FixedMassPlan(f["mesh"], cfg["L"], R; nbins=f["nbins"], stride=f["stride"], sub=Int.(f["sub"]),
                  to_device=dev, T=f["T_acc"] == "Float64" ? Float64 : Float32, level=f["level"])
end

function write_fm(h, cfg, fmres, epochs)
    h["epochs"] = epochs
    attrs(h)["config"] = sprint(TOML.print, delete!(copy(cfg), "config_fm")); attrs(h)["config_fm"] = cfg["config_fm"]
    attrs(h)["terms"] = collect(FM_TERMS); attrs(h)["channels"] = collect(FM_CHANNELS)
    for x in fmres
        g = create_group(h, "e$(x.e)_o$(x.order)")
        r = x.r
        g["R"] = r.R; g["edges"] = r.edges; g["hist"] = r.hist; g["mom"] = r.mom; g["tmom"] = r.tmom
        g["sub"] = r.sub; g["sub_R"] = r.sub_R
        attrs(g)["D"] = x.D; attrs(g)["order"] = x.order; attrs(g)["stride"] = r.stride
        for (k, v) in r.scal; attrs(g)[k] = v; end
    end
end

function run_job(cfg, j, pk, c)
    model, member, sL, sS, sg = j
    L = cfg["L"]; epochs = Float64.(cfg["epochs"]); fe = Float64.(cfg["fixedmass"]["epochs"])
    all(any(isapprox(D, x; atol=1e-9) for x in epochs) for D in fe) || error("fixedmass epochs must be base epochs")
    P = fm_plan(cfg, CuArray)
    if model == "lpt"
        lc = cfg["lpt"]; N = lc["N"]
        ms = ModeSet(N, L, cfg["k_store"]; to_device=CuArray)
        K = to_gpu(nlpt_kernels(N, L; T=Float32))
        f = CuArray(make_fphi(spec_of(cfg, N, sL, sS, sg), pk, member))
        t = @elapsed r = run_lpt(f, K, ms, L, epochs; n_max=lc["n_max"], mode=Symbol(get(lc, "mode", "fast")),
                                 fm=P, fm_orders=Int.(cfg["fixedmass"]["lpt_orders"]), fm_epochs=fe)
    else
        pc = cfg["pm"]; N = pc["N"]
        ms = ModeSet(N, L, cfg["k_store"]; to_device=CuArray)
        K = to_gpu(nlpt_kernels(N, L; T=Float32))
        f = CuArray(make_fphi(spec_of(cfg, N, sL, sS, sg), pk, member))
        Ψ0, Π0 = pm_ics(c, f, K, pc["D_ini"]; ic_order=pc["ic_order"], mode=Symbol(get(pc, "ic_mode", "fast")))
        f = nothing; K = nothing; GC.gc(true); CUDA.reclaim()
        st = (worder=get(pc, "worder", 2), grad_kernel_order=get(pc, "grad_kernel_order", 4),
              deconvolve=get(pc, "deconvolve", false), antialias=get(pc, "antialias", 0),
              n_resample=get(pc, "n_resample", 1), lean=get(pc, "lean", false))
        t = @elapsed r = run_pm(c, Ψ0, Π0, ms, L, epochs; D_ini=pc["D_ini"], n_steps=pc["n_steps"], res_pm=pc["res_pm"],
                                fm=P, fm_epochs=fe, st...)
    end
    h5open(fname(cfg, j), "w") do h
        write_fm(h, cfg, r.fm, epochs)
        attrs(h)["model"] = model; attrs(h)["member"] = String(member)
        attrs(h)["seed_L"] = sL; attrs(h)["seed_S"] = sS; attrs(h)["sign"] = sg
        attrs(h)["seconds"] = t; attrs(h)["date"] = string(now()); attrs(h)["gpu"] = CUDA.name(CUDA.device())
    end
end

function main(args)
    cfg = load_cfg(args[1]); model = length(args) >= 2 ? args[2] : "all"
    maxj = length(args) >= 3 ? parse(Int, args[3]) : typemax(Int)
    mkpath(joinpath(scratch(), cfg["name"]))
    cp(args[1], joinpath(scratch(), cfg["name"], "config.toml"); force=true)
    pk = planck_pk(); c = eds_cosmology()
    todo = filter(j -> !isfile(fname(cfg, j)), jobs(cfg, model))
    println("$(cfg["name"]): $(length(todo)) jobs to run"); flush(stdout)
    for j in todo[1:min(end, maxj)]
        t = @elapsed run_job(cfg, j, pk, c)
        @printf "%s  %-40s %6.1f s\n" Dates.format(now(), "HH:MM:SS") basename(fname(cfg, j)) t; flush(stdout)
        GC.gc(true); CUDA.reclaim()
    end
end
main(ARGS)
