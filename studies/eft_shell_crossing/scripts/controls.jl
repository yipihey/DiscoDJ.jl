# Initial-condition controls (CPU, LPT only): does the pair construction itself create a k² "response"?
#
#   julia --project=. -t 32 scripts/controls.jl          → results/controls.json
#
#   sharp      : the study's construction (shared |k| < k_split, band added)                    → C₂ = 0 at 1LPT
#   gauss      : PreCrossedSheet default — one seed, Gaussian filters R_UV, R_IR                 → C₂ = (R_UV²−R_IR²)/2 at 1LPT, for all D
#   gauss_mlv  : + "match linear variance" (IR amplitude rescaled by σ_UV/σ_IR)                  → adds a constant R = s − 1
#   null       : band amplitude 0 (UV ≡ IR)                                                     → exactly 0 at every order
# For each: binned R(k) and the k ≤ 0.08 fit of C₂ at 1LPT and 3LPT over the study's epochs.
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, DiscoDJNative, JSON3, Printf, Statistics
const DJ = DiscoDJNative

pk = planck_pk(); N = 256; L = 500.0; kf = 2π / L
base = PairSpec(N=N, N_ic=N, L=L, k_split=0.2, k_uv=1.2, seed_L=1, seed_S=1001)
RUV, RIR = 3.0, 6.0
epochs = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
K = nlpt_kernels(N, L; T=Float32)
ms = ModeSet(N, L, 0.199)
edges = collect(0.5:1:15.5) .* kf
cache = Dict()
shapes(f) = (sh = compute_core(f, K; n_order=3); Dict(n => vector_modes(ms, sh["psi_$n"]) for n in 1:3))

function sigma_gauss(R)      # continuum linear σ_δ at D = 1 of the Gaussian-filtered field (k < k_uv)
    k = exp.(range(log(1e-4), log(base.k_uv), length=4000))
    P = [DJ._jinterp(log(x), log.(pk["k"]), pk["Pk"]) for x in k] .* exp.(-k .^ 2 .* R^2)
    sqrt(sum(((k .^ 2 .* P)[2:end] .+ (k .^ 2 .* P)[1:end-1]) ./ 2 .* diff(k)) / (2π^2))
end
s_mlv = sigma_gauss(RUV) / sigma_gauss(RIR)

pairs = Dict(
    "sharp" => (make_fphi(base, pk, :UV; cache=cache), make_fphi(base, pk, :IR; cache=cache)),
    "null" => (make_fphi(PairSpec(base; band_amp=0.0), pk, :UV; cache=cache), make_fphi(base, pk, :IR; cache=cache)),
    "gauss" => (make_fphi_gauss(base, pk, RUV; cache=cache), make_fphi_gauss(base, pk, RIR; cache=cache)),
    "gauss_mlv" => (make_fphi_gauss(base, pk, RUV; cache=cache), Float32(s_mlv) .* make_fphi_gauss(base, pk, RIR; cache=cache)),
)
out = Dict{String,Any}("epochs" => epochs, "k" => Float64[], "R_UV" => RUV, "R_IR" => RIR, "s_mlv" => s_mlv,
                       "C2_gauss_1lpt_predicted" => (RUV^2 - RIR^2) / 2)
for (name, (fu, fi)) in pairs
    su = shapes(fu); si = shapes(fi)
    res = Dict{String,Any}()
    for n in (1, 3)
        C2 = Float64[]; Rk = Vector{Float64}[]
        for D in epochs
            Δ = sum(D^m .* (su[m] .- si[m]) for m in 1:n)
            x = D .* longitudinal(si[1], ms.kvec)              # regressor: the IR member's linear field
            A = accum!(longitudinal(Δ, ms.kvec), x, ms.kmag, edges); r = response(A)
            push!(C2, fit_counterterm(r.k2, r.k4, r.R, 1 ./ sqrt.(r.n); kmax=0.08).C2); push!(Rk, r.R)
            isempty(out["k"]) && (out["k"] = sqrt.(r.k2))
        end
        res["C2_$(n)lpt"] = C2; res["R_$(n)lpt"] = Rk
    end
    out[name] = res
    @printf "%-10s 1LPT C2(D) = %s\n           3LPT C2(D) = %s\n" name join([@sprintf("%9.4f", c) for c in res["C2_1lpt"]], " ") join([@sprintf("%9.4f", c) for c in res["C2_3lpt"]], " ")
end
@printf "predicted Gaussian-pair 1LPT C2 = (R_UV²−R_IR²)/2 = %.3f; mlv scale s = %.4f\n" (RUV^2 - RIR^2) / 2 s_mlv
open(io -> JSON3.write(io, out), joinpath(@__DIR__, "..", "results", "controls.json"), "w")
