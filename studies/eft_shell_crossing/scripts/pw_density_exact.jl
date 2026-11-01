# Deposit-bias calibration of the plane-wave density response (CPU, no mesh): exact δ(k_j) = N⁻³ Σ_q exp(−ik_j·(q+Ψ(q)))
# of 1/3/7LPT at the waves, ±ε, vs the stored CIC-deposited modes of the same run. → results/pw_density_exact.json
#
#   CUDA_VISIBLE_DEVICES="" julia --project=. -t 32 scripts/pw_density_exact.jl prod_ks20 prod_ks50 prod_ks70
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, DiscoDJNative, FFTW, Printf, Statistics, JSON3
FFTW.set_num_threads(min(Threads.nthreads(), 32))     # > 64 FFTW threads crash on this machine
scratch() = get(ENV, "EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
const Ds = (0.2, 0.3, 0.5, 0.7, 1.0); const ORD = (1, 3, 7)

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
    nvec = first(values(runs))["nvec"]; kvec = kf .* nvec; pw = cfg["planewaves"]
    pws = [Tuple(Int.(n)) for n in pw["n"]]
    idx = [findfirst(m -> Tuple(nvec[m, :]) == n || Tuple(nvec[m, :]) == .-n, axes(nvec, 1)) for n in pws]
    kv = kvec[idx, :]; k = vec(sqrt.(sum(abs2, kv; dims=2)))
    x1 = -im .* k .* longitudinal(linear_psi1_modes(cfg, 1, 1, nvec, pk)[idx, :], kv)
    K = nlpt_kernels(N, L; T=Float64); s = 1
    sh = Dict(sg => compute_core(make_fphi(PairSpec(N=N, N_ic=cfg["N_ic"], L=L, k_split=cfg["k_split"], k_uv=cfg["k_uv"], seed_L=s, seed_S=s,
                                                    long_sign=sg, pw=pws, pw_amp=pw["amp"]), pk, :UV; T=Float64), K; n_order=7) for sg in (1, -1))
    res = Any[]
    @printf "\n%s (realization 1): R(deposit) − R(exact) fitted to −b k²; C₂ exact vs deposit (k²-only, k ≤ 0.1)\n" name
    lo = k .<= 0.1; A = reshape(-k[lo] .^ 2, :, 1)
    for D in Ds, n in ORD
        e = findfirst(≈(D), E)
        field(sg) = sum(D^m .* sh[sg]["psi_$m"] for m in 1:n)
        δodd = (exact_modes(field(1), kv, L) .- exact_modes(field(-1), kv, L)) ./ 2
        x = D .* x1
        Rex = real.((δodd .- x) .* conj.(x)) ./ abs2.(x)
        dep(sg) = EFTShell._fields(runs[("lpt", :UV, s, s, sg)], "lpt$n", e, D)[3][idx]
        Rdep = real.(((dep(1) .- dep(-1)) ./ 2 .- x) .* conj.(x)) ./ abs2.(x)
        cex = (A \ Rex[lo])[1]; cdp = (A \ Rdep[lo])[1]
        @printf "D=%.1f  %dLPT  C₂ exact %+8.4f  deposit %+8.4f  bias %+8.4f  (per-wave rms of the k²-fit residual of the bias %.1e)\n" D n cex cdp cdp - cex sqrt(mean(abs2, (Rdep .- Rex)[lo] .- A * [cdp - cex]))
        push!(res, Dict("D" => D, "n" => n, "C2_exact" => cex, "C2_deposit" => cdp, "k" => k, "R_exact" => Rex, "R_deposit" => Rdep))
    end
    out[name] = res
    open(io -> JSON3.write(io, out), joinpath(@__DIR__, "..", "results", "pw_density_exact.json"), "w")
end
