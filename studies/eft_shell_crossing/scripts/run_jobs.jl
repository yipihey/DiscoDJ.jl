# Run every missing (model, member, seed) job of a configuration; one compact HDF5 file per job.
#
#   julia --project=. scripts/run_jobs.jl config/<name>.toml [lpt|pm|all] [max_jobs]
#
# Output: $EFTSC_SCRATCH/<name>/<model>_<UV|IR>_L<seedL>_S<seedS>_s<±1>.h5  (seedS = 0 for IR).
# Restartable: existing files are skipped, so a crashed/OOM process can simply be relaunched.
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, CUDA, DiscoDJNative, HDF5, TOML, Printf, Dates
const DJ = DiscoDJNative

scratch() = get(ENV, "EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")

# [seeds] long, short, signs; optional extra = [{long = 1, short = [...]}, ...] adds short seeds for
# particular long seeds (conditional ensembles at fixed long modes, Experiment B).
function shorts_of(sd, sL)
    get(sd, "single_field", false) && return [sL]    # band drawn from the long seed's own white noise:
                                                      # the UV member is then the same for every k_split
    s = collect(sd["short"])
    for g in get(sd, "extra", [])
        g["long"] == sL && append!(s, g["short"])
    end
    return s
end
function jobs(cfg, model)
    sd = cfg["seeds"]; out = []
    members = Symbol.(get(sd, "members", ["IR", "UV"]))
    for m in (model == "all" ? ("lpt", "pm") : (model,)), sL in sd["long"], sg in sd["signs"]
        :IR in members && push!(out, (m, :IR, sL, 0, sg))
        :UV in members || continue
        for sS in shorts_of(sd, sL)
            push!(out, (m, :UV, sL, sS, sg))
        end
    end
    return out
end
fname(cfg, j) = joinpath(scratch(), cfg["name"], @sprintf("%s_%s_L%d_S%d_s%+d.h5", j[1], j[2], j[3], j[4], j[5]))

cis_plan(cfg, N, dev) = haskey(cfg, "cis") ?
    CiSPlan(N, cfg["L"], Float64.(cfg["cis"]["R"]); stride=get(cfg["cis"], "stride", 8), to_device=dev,
            kcut=Float64.(get(cfg["cis"], "kcuts", [cfg["k_split"]]))) : nothing
# experiment 2: kinetic-stress meshes (ModeSets on each coarse-graining mesh)
stress_sets(cfg, dev) = haskey(cfg, "stress") ?
    [ModeSet(Nm, cfg["L"], cfg["k_store"]; to_device=dev) for Nm in cfg["stress"]["meshes"]] : nothing
cis_epochs(cfg) = haskey(cfg, "cis") ? Float64.(get(cfg["cis"], "epochs", cfg["epochs"])) : Float64[]

# sign codes: ±1 = all plane waves ×(±1); ±2 = group A ×(±1), group B (planewaves.group_b, 1-based) ×(∓1)
_pwn(cfg) = [Tuple(Int.(n)) for n in get(get(cfg, "planewaves", Dict()), "n", [])]
_pwflip(cfg, sg) = abs(sg) == 2 ? _pwn(cfg)[Int.(cfg["planewaves"]["group_b"])] : NTuple{3,Int}[]
spec_of(cfg, N, sL, sS, sg) = PairSpec(N=N, N_ic=cfg["N_ic"], L=cfg["L"], k_split=cfg["k_split"], k_uv=cfg["k_uv"],
                                       k_lo=get(cfg, "k_lo", 0.0), band_amp=get(cfg, "band_amp", 1.0),
                                       long_sign=Int(sign(sg)), seed_L=sL, seed_S=sS == 0 ? 1 : sS,
                                       pw=_pwn(cfg), pw_flip=_pwflip(cfg, sg),
                                       pw_amp=get(get(cfg, "planewaves", Dict()), "amp", 0.0),
                                       pw_bg=get(get(cfg, "planewaves", Dict()), "background", 0.0))

function write_common(f, cfg, ms, j)
    f["nvec"] = ms.nvec; f["epochs"] = Float64.(cfg["epochs"])
    attrs(f)["config"] = sprint(TOML.print, cfg)
    attrs(f)["model"] = j[1]; attrs(f)["member"] = String(j[2])
    attrs(f)["seed_L"] = j[3]; attrs(f)["seed_S"] = j[4]; attrs(f)["sign"] = j[5]
    attrs(f)["date"] = string(now()); attrs(f)["gpu"] = CUDA.functional() ? CUDA.name(CUDA.device()) : "none"
    attrs(f)["threads"] = Threads.nthreads()
end

function run_job(cfg, j, pk, c)
    model, member, sL, sS, sg = j
    L = cfg["L"]; epochs = Float64.(cfg["epochs"])
    if model == "lpt"
        lc = cfg["lpt"]; N = lc["N"]
        ms = ModeSet(N, L, cfg["k_store"]; to_device=CuArray)
        msb = (haskey(lc, "band_k") && member == :UV) ? ModeSet(N, L, lc["band_k"]; to_device=CuArray) : nothing
        K = to_gpu(nlpt_kernels(N, L; T=Float32))
        f = CuArray(make_fphi(spec_of(cfg, N, sL, sS, sg), pk, member))
        cp_ = cis_plan(cfg, N, CuArray)
        t = @elapsed r = run_lpt(f, K, ms, L, epochs; n_max=lc["n_max"], mode=Symbol(get(lc, "mode", "fast")), ms_band=msb,
                                 cis=cp_, cis_orders=haskey(cfg, "cis") ? Int.(cfg["cis"]["lpt_orders"]) : Int[],
                                 cis_epochs=cis_epochs(cfg), rsd=get(cfg, "rsd", false),
                                 fgrowth=D -> DJ.growth_Dplusda(c, DJ.a_of_Dplus(c, D)) * DJ.a_of_Dplus(c, D) / D,
                                 stress=stress_sets(cfg, CuArray), stress_order=haskey(cfg, "stress") ? cfg["stress"]["lpt_order"] : 0)
        h5open(fname(cfg, j), "w") do h
            write_common(h, cfg, ms, j)
            h["psi"] = ComplexF32.(r.psi); h["dens"] = ComplexF32.(r.dens); h["inv"] = r.inv; h["rms"] = r.rms
            r.dens_s === nothing || (h["dens_rsd"] = ComplexF32.(r.dens_s); h["inv_rsd"] = r.inv_s)
            r.stress === nothing || (h["stress"] = ComplexF32.(r.stress); h["stress_scalars"] = r.stress_scalars;
                                     h["stress_meshes"] = Int.(cfg["stress"]["meshes"]))
            if msb !== nothing
                h["band_nvec"] = msb.nvec; h["band_psi"] = ComplexF32.(r.band)
            end
            if cp_ !== nothing
                h["cis"] = r.cis; h["cis_orders"] = r.cis_orders; h["cis_R"] = cp_.R; h["cis_lin"] = cis_linear(cp_, f)
                h["cis_kcut"] = cp_.kcut; h["cis_epochs"] = cis_epochs(cfg)
            end
            attrs(h)["seconds"] = t
        end
    else
        pc = cfg["pm"]; N = pc["N"]
        cpu = get(pc, "device", "gpu") == "cpu"          # large PM meshes that do not fit the GPU
        dev = cpu ? identity : CuArray
        ms = ModeSet(N, L, cfg["k_store"]; to_device=dev)
        band = (haskey(pc, "band_k") && member == :UV) ?
               (ModeSet(N, L, pc["band_k"]; to_device=dev), Float64.(pc["band_epochs"])) : nothing
        K = cpu ? nlpt_kernels(N, L; T=Float32) : to_gpu(nlpt_kernels(N, L; T=Float32))
        f = dev(make_fphi(spec_of(cfg, N, sL, sS, sg), pk, member))
        cp_ = cis_plan(cfg, N, dev); cl_ = cp_ === nothing ? nothing : cis_linear(cp_, f)
        Ψ0, Π0 = pm_ics(c, f, K, pc["D_ini"]; ic_order=pc["ic_order"], mode=Symbol(get(pc, "ic_mode", "fast")))
        f = nothing; K = nothing; GC.gc(true); cpu || CUDA.reclaim()
        st = (worder=get(pc, "worder", 2), grad_kernel_order=get(pc, "grad_kernel_order", 4),
              deconvolve=get(pc, "deconvolve", false), antialias=get(pc, "antialias", 0),
              n_resample=get(pc, "n_resample", 1), lean=get(pc, "lean", false))
        r = run_pm(c, Ψ0, Π0, ms, L, epochs; D_ini=pc["D_ini"], n_steps=pc["n_steps"], res_pm=pc["res_pm"],
                   band=band, cis=cp_, cis_epochs=cis_epochs(cfg), rsd=get(cfg, "rsd", false),
                   stress=stress_sets(cfg, dev),
                   pkplan=haskey(cfg, "pk_full") ? PkPlan(L, cfg["pk_full"]["mesh"]; nbins=cfg["pk_full"]["nbins"]) : nothing, st...)
        h5open(fname(cfg, j), "w") do h
            write_common(h, cfg, ms, j)
            h["psi"] = ComplexF32.(r.psi); h["vel"] = ComplexF32.(r.vel); h["dens"] = ComplexF32.(r.dens)
            h["inv"] = r.inv; h["rms"] = r.rms; h["mom"] = r.mom; h["grid"] = r.grid
            r.dens_s === nothing || (h["dens_rsd"] = ComplexF32.(r.dens_s); h["inv_rsd"] = r.inv_s)
            r.stress === nothing || (h["stress"] = ComplexF32.(r.stress); h["stress_scalars"] = r.stress_scalars;
                                     h["stress_meshes"] = Int.(cfg["stress"]["meshes"]))
            if r.pk_dens !== nothing
                pp = PkPlan(L, cfg["pk_full"]["mesh"]; nbins=cfg["pk_full"]["nbins"])
                h["pk_k"] = sqrt.(pp.edges[1:end-1] .* pp.edges[2:end]); h["pk_dens"] = r.pk_dens; h["pk_disp"] = r.pk_disp
            end
            if band !== nothing
                h["band_nvec"] = band[1].nvec; h["band_epochs"] = band[2]; h["band_psi"] = ComplexF32.(r.band)
            end
            if cp_ !== nothing
                h["cis"] = r.cis; h["cis_R"] = cp_.R; h["cis_lin"] = cl_
                h["cis_kcut"] = cp_.kcut; h["cis_epochs"] = cis_epochs(cfg)
            end
            attrs(h)["seconds"] = r.seconds
        end
    end
end

function main(args)
    cfg = TOML.parsefile(args[1]); model = length(args) >= 2 ? args[2] : "all"
    maxj = length(args) >= 3 ? parse(Int, args[3]) : typemax(Int)
    mkpath(joinpath(scratch(), cfg["name"]))
    cp(args[1], joinpath(scratch(), cfg["name"], "config.toml"); force=true)
    pk = planck_pk(); c = eds_cosmology()
    # variants with the same long modes and solver settings share the IR members of another config
    for (key, member) in (("ir_from", :IR), ("uv_from", :UV))
        haskey(cfg, key) || continue
        for j in jobs(cfg, model)
            j[2] === member || continue
            src = joinpath(scratch(), cfg[key], basename(fname(cfg, j)))
            (isfile(src) && !islink(fname(cfg, j)) && !isfile(fname(cfg, j))) && symlink(src, fname(cfg, j))
        end
    end
    todo = filter(j -> !isfile(fname(cfg, j)), jobs(cfg, model))
    println("$(cfg["name"]): $(length(todo)) jobs to run"); flush(stdout)
    for j in todo[1:min(end, maxj)]
        t = @elapsed run_job(cfg, j, pk, c)
        @printf "%s  %-40s %6.1f s\n" Dates.format(now(), "HH:MM:SS") basename(fname(cfg, j)) t; flush(stdout)
        GC.gc(true); CUDA.reclaim()
    end
end
main(ARGS)
