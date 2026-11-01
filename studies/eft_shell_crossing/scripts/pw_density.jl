# Eulerian-density plane-wave response → results/pw_density.json          (CPU; reads the stored low-k density modes)
#
#   julia --project=. scripts/pw_density.jl prod_ks12 ... prod_ks20_bg05 prod_ks20_bg10
#
# R^δ_j = Re(Δ_j x_j*)/|x_j|²,  Δ_j = [δ(+ε) − δ(−ε)]/2 − ref_j,  x_j = −i k_j x_∥,j  (linear density of the waves).
# ref_j = the band-free member's odd density where it was run (long background), else the linear x_j.  Fits over the
# waves, jackknife over realizations: R = −C₂k² + C₄k⁴ (default), R = −C₂k² on k ≤ 0.1, and R = R₀ − C₂k² + C₄k⁴.
# 1LPT is purely kinematic and known exactly: R_ZA(k) = exp(−½D² k·Σ·k) − 1, Σ = ⟨ψψ⟩ of the realization's own band
# (→ C₂^δ = σ²_ψ D²/6).  The CIC deposit has a coherent bias on near-lattice loads (≈ −0.18k²), so the measured 1LPT
# differs from this at early times.  "clean" response: R_m − R_1LPT + R_ZA (same deposit, same particles ⇒ bias cancels).
# results/pw_density_perseed.json keeps per-realization arrays for matched differences (model, epoch, wave, seed):
#   R    relative to the bare wave x (the fits above)      Rref  relative to the band-free member (= R when that member is x)
#   Rv   Lagrangian velocity response, same reference      RZA   kinematic R_ZA of the realization's band
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, Printf, Statistics, JSON3
scratch() = get(ENV, "EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
const MODELS = ["pm", "lpt1", "lpt3", "lpt5", "lpt7"]
"""⟨ψ_iψ_j⟩ at D = 1 of the band (UV − IR potential) of realization s, on the box's own modes."""
function band_tensor(cfg, s, pk, cache)
    pw = cfg["planewaves"]; N = cfg["N_ic"]; L = cfg["L"]
    sp = PairSpec(N=N, N_ic=N, L=L, k_split=cfg["k_split"], k_uv=cfg["k_uv"], seed_L=s, seed_S=s,
                  pw=[Tuple(Int.(n)) for n in pw["n"]], pw_amp=pw["amp"], pw_bg=get(pw, "background", 0.0))
    φ = make_fphi(sp, pk, :UV; T=Float64, cache=cache) .- make_fphi(sp, pk, :IR; T=Float64, cache=cache)
    Σ = zeros(3, 3); kf = 2π / L; sg(m) = m < N ÷ 2 ? m : m - N
    @inbounds for l in 1:N÷2+1, j in 1:N, i in 1:N
        a = abs2(φ[i, j, l]); a == 0 && continue
        v = kf .* (sg(i - 1), sg(j - 1), l - 1); w = (l == 1 || l == N ÷ 2 + 1) ? 1.0 : 2.0
        for b in 1:3, c in 1:3; Σ[b, c] += w * v[b] * v[c] * a / N^6; end
    end
    return Σ
end
jk(f, X) = begin   # X: (wave, seed)
    n = size(X, 2); full = f(vec(mean(X; dims=2)))
    parts = [f(vec(mean(X[:, setdiff(1:n, i)]; dims=2))) for i in 1:n]; m = sum(parts) ./ n
    full, sqrt.((n - 1) / n .* sum(p -> (p .- m) .^ 2, parts))
end
nest3(A) = [[[isfinite(A[e, j, i]) ? A[e, j, i] : nothing for i in axes(A, 3)] for j in axes(A, 2)] for e in axes(A, 1)]   # JSON has no NaN (PM before D_ini)
out = Dict{String,Any}(); perseed = Dict{String,Any}()
for name in ARGS
    cfg, runs = load_config(joinpath(scratch(), name)); pk = planck_pk()
    kf = 2π / cfg["L"]; E = Float64.(cfg["epochs"]); Dini = cfg["pm"]["D_ini"]
    seeds = sort(unique(k[3] for k in keys(runs)))
    nvec = first(values(runs))["nvec"]; kvec = kf .* nvec
    pws = [Tuple(Int.(n)) for n in cfg["planewaves"]["n"]]
    idx = [findfirst(m -> Tuple(nvec[m, :]) == n || Tuple(nvec[m, :]) == .-n, axes(nvec, 1)) for n in pws]
    kv = kvec[idx, :]; k = vec(sqrt.(sum(abs2, kv; dims=2))); o = sortperm(k); k = k[o]
    ψ1 = linear_psi1_modes(cfg, 1, 1, nvec, pk)[idx, :]
    res = Dict{String,Any}("k" => k, "epochs" => Float64[], "k_split" => cfg["k_split"], "models" => Dict{String,Any}())
    R = Dict(m => fill(NaN, length(E), length(k), length(seeds)) for m in MODELS)
    Rref = Dict(m => fill(NaN, length(E), length(k), length(seeds)) for m in MODELS); Rv = deepcopy(Rref)
    kvs = kv[o, :]; RZA = fill(NaN, length(E), length(k), length(seeds)); cache = Dict(); sig2 = Float64[]
    for (is, s) in enumerate(seeds)
        Σ = band_tensor(cfg, s, pk, cache); push!(sig2, Σ[1, 1] + Σ[2, 2] + Σ[3, 3])
        for (e, D) in enumerate(E), j in eachindex(k)
            q = kvs[j, :]; RZA[e, j, is] = expm1(-0.5 * D^2 * sum(q .* (Σ * q)))
        end
    end
    res["sigma2_psi_band"] = sig2
    for (is, s) in enumerate(seeds), m in MODELS, (e, D) in enumerate(E)
        mt = m == "pm" ? "pm" : "lpt"; (m == "pm" && D <= Dini) && continue
        up = get(runs, (mt, :UV, s, s, 1), nothing); um = get(runs, (mt, :UV, s, s, -1), nothing)
        (up === nothing || um === nothing) && continue
        dens(r) = EFTShell._fields(r, m, e, D)[3][idx]
        x = -im .* vec(sqrt.(sum(abs2, kv; dims=2))) .* (D .* longitudinal(ψ1, kv))
        ip = get(runs, (mt, :IR, s, 0, 1), nothing); imn = get(runs, (mt, :IR, s, 0, -1), nothing)
        ref = (ip === nothing || imn === nothing) ? x : (dens(ip) .- dens(imn)) ./ 2
        Δ = (dens(up) .- dens(um)) ./ 2 .- ref
        R[m][e, :, is] .= (real.(Δ .* conj.(x)) ./ abs2.(x))[o]
        Rref[m][e, :, is] .= (real.(Δ .* conj.(ref)) ./ abs2.(ref))[o]
        vel(r) = EFTShell._fields(r, m, e, D)[2][idx, :]
        xv = longitudinal(ψ1, kv)
        vref = (ip === nothing || imn === nothing) ? xv : longitudinal((vel(ip) .- vel(imn)) ./ 2, kv)
        Δv = longitudinal((vel(up) .- vel(um)) ./ 2, kv) .- vref
        Rv[m][e, :, is] .= (real.(Δv .* conj.(xv)) ./ abs2.(xv))[o]
    end
    A2 = hcat(-k .^ 2, k .^ 4); A0 = hcat(ones(length(k)), -k .^ 2, k .^ 4); lo = k .<= 0.1; A1 = reshape(-k[lo] .^ 2, :, 1)
    for m in MODELS
        rows = Any[]
        for (e, D) in enumerate(E)
            all(isfinite, R[m][e, :, :]) || continue
            X = R[m][e, :, :]
            v2, s2 = jk(r -> A2 \ r, X); v1, s1 = jk(r -> A1 \ r[lo], X); v0, s0 = jk(r -> A0 \ r, X)
            Xc = X .- R["lpt1"][e, :, :] .+ RZA[e, :, :]
            vc, sc = jk(r -> A2 \ r, Xc); vk, sk = jk(r -> A1 \ r[lo], Xc)
            push!(rows, Dict("D" => D, "C2_clean" => vc[1], "sC2_clean" => sc[1], "C4_clean" => vc[2], "sC4_clean" => sc[2],
                             "C2_clean_k2only" => vk[1], "sC2_clean_k2only" => sk[1], "C2" => v2[1], "sC2" => s2[1], "C4" => v2[2], "sC4" => s2[2], "C2_k2only" => v1[1], "sC2_k2only" => s1[1],
                             "R0" => v0[1], "sR0" => s0[1], "C2_r0" => v0[2], "sC2_r0" => s0[2], "C4_r0" => v0[3], "sC4_r0" => s0[3]))
        end
        res["models"][m] = rows
    end
    out[name] = res
    perseed[name] = Dict("k" => k, "epochs" => E, "seeds" => seeds, "RZA" => nest3(RZA),
                         "R" => Dict(m => nest3(R[m]) for m in MODELS), "Rref" => Dict(m => nest3(Rref[m]) for m in MODELS),
                         "Rv" => Dict(m => nest3(Rv[m]) for m in MODELS))
    kin = sum(sig2) / length(sig2) / 6
    @printf "\n%s  k_split = %.2f, a = %.1f   σ²_ψ,band/6 = %.3f (realized)\n" name cfg["k_split"] get(cfg["planewaves"], "background", 0.0) kin
    println("   D  | deposit bias: C₂(1LPT, k²-only) − σ²D²/6 | clean C₂δ/D², 2-term: 3LPT   7LPT   PM ± σ    | PM k²-only ± σ | PM/kinematic | ΔP/P at k=0.05")
    for D in (0.2, 0.3, 0.5, 0.7, 1.0)
        g(m) = only(r for r in res["models"][m] if r["D"] ≈ D); p = g("pm")
        @printf "%5.2f | %+8.4f | %7.3f %8.3f %7.3f ± %.3f | %7.3f ± %.3f | %.3f | %+.2f%%\n" D g("lpt1")["C2_k2only"] - kin * D^2 g("lpt3")["C2_clean"] / D^2 g("lpt7")["C2_clean"] / D^2 p["C2_clean"] / D^2 p["sC2_clean"] / D^2 p["C2_clean_k2only"] / D^2 p["sC2_clean_k2only"] / D^2 p["C2_clean"] / D^2 / kin -200 * p["C2_clean"] * 0.05^2
    end
end
open(io -> JSON3.write(io, out), joinpath(@__DIR__, "..", "results", "pw_density.json"), "w")
open(io -> JSON3.write(io, perseed), joinpath(@__DIR__, "..", "results", "pw_density_perseed.json"), "w")
