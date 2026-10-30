# k⁴ convergence: realization-level ΔC2, ΔC4 of (R_A − R_B), order-2 fit (k ≤ 0.18), fixed mode-count weights,
# for successive LPT orders and for PM − 7LPT at several PM resolutions (single realization L1/S1001).
#   julia --project=. scripts/c4_convergence.jl conv_pm256m2bf conv_pm384m2bf conv_pm512m2lean > results/c4_convergence.txt
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, Printf
scr = get(ENV, "EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
cl, rl = load_config(joinpath(scr, "conv_lpt256")); pk = planck_pk()
L = cl["L"]; kf = 2π / L; edges = collect(0.5:1:(cl["k_split"] / kf)) .* kf
key(r, m, w) = only(k for k in keys(r) if k[1] == m && k[2] == w && k[3] == 1 && k[5] == 1)
nvec = rl[key(rl, "lpt", :IR)]["nvec"]; kvec = kf .* nvec; km = vec(sqrt.(sum(abs2, kvec; dims=2)))
sel = km .< cl["k_split"]; kvec = kvec[sel, :]; km = km[sel]
ψ1 = linear_psi1_modes(cl, 1, 1, nvec, pk)[sel, :]; epochs = Float64.(cl["epochs"])
lptΔ(n, D) = sum(D^m .* ComplexF64.(rl[key(rl, "lpt", :UV)]["psi"][m, :, :] .- rl[key(rl, "lpt", :IR)]["psi"][m, :, :]) for m in 1:n)[sel, :]
function fit2(y, x; kmax=0.18)
    A = accum!(y, x, km, edges); r = response(A)
    σ = sqrt.(max.(A.yy .- r.R .* A.yx, 0) ./ A.n ./ (2 .* A.xx))
    w = 1 ./ sqrt.(A.n)           # fixed weights (mode counting), scaled to the residual level
    w = w .* (sum(σ) / sum(w))
    f = fit_counterterm(r.k2, r.k4, r.R, w; kmax=kmax, order=2); f
end
pms = ARGS
println("ΔC2, ΔC4 [(Mpc/h)^2, (Mpc/h)^4] of realization-level differences, order-2 fit, k ≤ 0.18, fixed weights")
@printf "%6s  %-22s %-22s | %s\n" "D" "5LPT−3LPT" "7LPT−5LPT" join(["PM−7LPT " * p for p in pms], " | ")
pmruns = [load_config(joinpath(scr, p))[2] for p in pms]
for (e, D) in enumerate(epochs)
    D < 0.1 && continue
    x = D .* longitudinal(ψ1, kvec)
    d53 = fit2(longitudinal(lptΔ(5, D) .- lptΔ(3, D), kvec), x); d75 = fit2(longitudinal(lptΔ(7, D) .- lptΔ(5, D), kvec), x)
    @printf "%6.3f  %+8.4f %+9.3f      %+8.4f %+9.3f     |" D d53.C2 d53.C4 d75.C2 d75.C4
    for rp in pmruns
        dpm = ComplexF64.(rp[key(rp, "pm", :UV)]["psi"][e, :, :] .- rp[key(rp, "pm", :IR)]["psi"][e, :, :])[sel, :]
        f = fit2(longitudinal(dpm .- lptΔ(7, D), kvec), x)
        @printf " %+8.4f ± %.4f  %+8.3f ± %.3f |" f.C2 f.σC2 f.C4 f.σC4
    end
    println()
end
