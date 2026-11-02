# Finite-ε response of the plane waves alone (no band): direct-sum density of 1LPT and 7LPT → results/pw_density_waves.json
#
#   CUDA_VISIBLE_DEVICES="" julia --project=. -t 8 scripts/pw_waves_only.jl prod_ks12 ... prod_ks70
#
# The ± estimator cancels even orders in ε but keeps ε³, so at finite ε the waves smear one another:
#   R_j = −D²ε² [ Σ_{m≠j} (k_j·k̂_m)²/(4k_m²) + 1/8 ],   the last term being the wave's own Bessel J₁ correction.
# This is the reference to subtract from the density response at a = 0 (README §14.11).  Waves are sorted by |k|, as in
# results/pw_density_perseed.json.
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, DiscoDJNative, FFTW, Printf, JSON3
FFTW.set_num_threads(min(Threads.nthreads(), 32))     # > 64 FFTW threads crash on this machine
scratch() = get(ENV, "EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")

function exact_modes(Ψ, kv, L)
    N = size(Ψ, 1); out = zeros(ComplexF64, size(kv, 1)); dq = L / N
    for j in axes(kv, 1)
        kx, ky, kz = kv[j, 1], kv[j, 2], kv[j, 3]
        parts = zeros(ComplexF64, N)
        Threads.@threads for l in 1:N
            s = 0.0 + 0.0im
            @inbounds for b in 1:N, a in 1:N
                ph = kx * ((a - 1) * dq + Ψ[a, b, l, 1]) + ky * ((b - 1) * dq + Ψ[a, b, l, 2]) + kz * ((l - 1) * dq + Ψ[a, b, l, 3])
                s += cis(-ph)
            end
            parts[l] = s
        end
        out[j] = sum(parts) / N^3
    end
    return out
end

out = Dict{String,Any}(); pk = planck_pk()
for name in ARGS
    cfg, runs = load_config(joinpath(scratch(), name)); L = cfg["L"]; N = cfg["lpt"]["N"]; kf = 2π / L; E = Float64.(cfg["epochs"])
    nvec = first(values(runs))["nvec"]; pw = cfg["planewaves"]; pws = [Tuple(Int.(n)) for n in pw["n"]]
    idx = [findfirst(m -> Tuple(nvec[m, :]) == n || Tuple(nvec[m, :]) == .-n, axes(nvec, 1)) for n in pws]
    kv = (kf .* nvec)[idx, :]; k = vec(sqrt.(sum(abs2, kv; dims=2))); o = sortperm(k)
    x1 = -im .* k .* longitudinal(linear_psi1_modes(cfg, 1, 1, nvec, pk)[idx, :], kv)
    K = nlpt_kernels(N, L; T=Float64)
    sh = Dict(sg => compute_core(make_fphi(PairSpec(N=N, N_ic=cfg["N_ic"], L=L, k_split=cfg["k_split"], k_uv=cfg["k_uv"], seed_L=1, seed_S=1,
                                                    long_sign=sg, pw=pws, pw_amp=pw["amp"]), pk, :IR; T=Float64), K; n_order=7) for sg in (1, -1))
    res = Dict{String,Any}("k" => k[o], "epochs" => E, "amp" => pw["amp"], "n" => Dict{String,Any}())
    for n in (1, 7)
        R = zeros(length(E), length(k))
        for (e, D) in enumerate(E)
            field(sg) = sum(D^m .* sh[sg]["psi_$m"] for m in 1:n)
            δodd = (exact_modes(field(1), kv, L) .- exact_modes(field(-1), kv, L)) ./ 2
            x = D .* x1
            R[e, :] .= (real.((δodd .- x) .* conj.(x)) ./ abs2.(x))[o]
        end
        res["n"][string(n)] = [R[e, :] for e in axes(R, 1)]
    end
    out[name] = res
    @printf "%s done: R(7LPT, D=1) lowest wave %.3e, highest %.3e\n" name res["n"]["7"][end][1] res["n"]["7"][end][end]; flush(stdout)
    open(io -> JSON3.write(io, out), joinpath(@__DIR__, "..", "results", "pw_density_waves.json"), "w")
    sh = nothing; GC.gc()
end
println("WAVES DONE")
