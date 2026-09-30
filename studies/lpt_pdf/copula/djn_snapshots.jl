# Displacement snapshots for the copula study computed with DiscoDJNative (Julia):
# nLPT from compute_core_exact / compute_core, N-body from the faithful port of DISCO-DJ run_nbody.
#
#   julia -t auto --project=<this directory> djn_snapshots.jl <fphi_base> <N> <scratch> <models> [seed_suffix]
#
#   fphi_base  written by `python snapshots.py fphi N` (<base>_re.npy, <base>_im.npy)
#   models     comma list of 1lpt,2lpt,3lpt,4lpt,nbody
#   env DJN_ZLIST=0,1  restricts the snapshot redshifts (default: config.json z_list)
# Writes psi_<model>_N<N>_z<z>[_seedS].npy (numpy (3,N,N,N) float64, x = q + psi) for every z in
# config.json, and a JSON log of the settings next to them.
using DiscoDJNative, JSON3

include(joinpath(@__DIR__, "npy.jl"))

function main(args)
    base, N, scratch, models = args[1], parse(Int, args[2]), args[3], split(args[4], ",")
    suffix = length(args) >= 5 ? args[5] : ""
    cfg = JSON3.read(read(joinpath(@__DIR__, "config.json"), String))
    L = Float64(cfg.box_L); ai = Float64(cfg.a_initial)
    # DJN_ZLIST (e.g. "0") overrides the snapshot redshifts of config.json
    zs = haskey(ENV, "DJN_ZLIST") ? parse.(Float64, split(ENV["DJN_ZLIST"], ",")) : Float64.(cfg.z_list)
    c = Cosmology("Planck18EEBAOSN")
    fphi = permutedims(read_npy(base * "_re.npy") .+ im .* read_npy(base * "_im.npy"), (3, 2, 1))
    t0 = time()
    need_eds = any(m -> m == "4lpt", models)
    shx = compute_core_exact(fphi, nlpt_kernels(N, L); n_order=3)        # exact growth ≤ 3rd order
    she = need_eds ? compute_core(fphi, nlpt_kernels(N, L); n_order=4) : nothing
    println("nLPT shapes N=$N: $(round(time() - t0, digits=1)) s"); flush(stdout)
    D(a) = DiscoDJNative._Dplus(c, a)
    g3(a) = [DiscoDJNative._jinterp(a, c._a_table, t) for t in (c._D3a_table, c._D3b_table, c._D3c_table)]
    lpt(m, a) = begin
        ψ = D(a) .* shx["psi_1"]
        m == "1lpt" && return ψ
        ψ .+= DiscoDJNative._D2plus(c, a) .* shx["psi_2_ex"]
        m == "2lpt" && return ψ
        d3 = g3(a)
        ψ .+= d3[1] .* shx["psi_3a_ex"] .+ d3[2] .* shx["psi_3b_ex"] .+ d3[3] .* shx["psi_3c_ex"]
        m == "3lpt" && return ψ
        ψ .+= D(a)^4 .* she["psi_4"]
        return ψ
    end
    fname(m, z) = joinpath(scratch, "psi_$(m)_N$(N)_z$(z == round(z) ? Int(z) : z)$(suffix).npy")
    for m in models
        m == "nbody" && continue
        for z in zs
            write_npy(fname(m, z), permutedims(lpt(m, 1 / (1 + z)), (3, 2, 1, 4)))
        end
        println("$m N=$N saved"); flush(stdout)
    end
    if "nbody" in models
        nb = cfg.nbody_djn
        # BullFrog in D-time, a grid uniform in D from a_initial to 1 with every snapshot a as a node
        ns = Int(nb.steps); Ds = collect(range(D(ai), D(1.0), length=ns + 1))
        for z in zs
            a = 1 / (1 + z); i = argmin(abs.(Ds .- D(a))); Ds[i] = D(a)
        end
        agrid = [DiscoDJNative.a_of_Dplus(c, d) for d in Ds]
        agrid[1] = ai; agrid[end] = 1.0
        snap = Dict(round(1 / (1 + z), digits=10) => z for z in zs)
        Ψ0, Π0 = nbody_ics_lpt(c, shx, ai; n_order=2, exact_growth=true)
        q1 = [(i - 1) * L / N for i in 1:N]
        function cb(k, a, X, P)
            key = round(a, digits=10)
            haskey(snap, key) || return
            ψ = copy(X)
            for (d, sh) in enumerate(((N, 1, 1), (1, N, 1), (1, 1, N)))
                ψ[:, :, :, d] .-= reshape(q1, sh)
            end
            write_npy(fname("nbody", snap[key]), permutedims(ψ, (3, 2, 1, 4)))
            println("nbody N=$N z=$(snap[key]) saved ($(round(time() - t0, digits=0)) s, step $k)"); flush(stdout)
        end
        run_nbody(c, Ψ0, Π0; boxsize=L, a_ini=ai, a_end=1.0, n_steps=nothing, time_var=agrid,
                  res_pm=Int(nb.mesh_factor) * N, stepper=Symbol(nb.stepper),
                  grad_kernel_order=Int(nb.grad_kernel_order), worder=Int(nb.worder),
                  deconvolve=Bool(nb.deconvolve), antialias=Int(nb.antialias), step_callback=cb)
    end
    open(joinpath(scratch, "djn_snapshots_N$(N)$(suffix).json"), "w") do io
        JSON3.write(io, Dict("N" => N, "models" => models, "nbody" => cfg.nbody_djn, "lpt" =>
            "compute_core_exact(n_order=3) + compute_core(n_order=4) psi_4, de-aliased (ext=3N/2)",
            "seconds" => time() - t0))
    end
end

main(ARGS)
