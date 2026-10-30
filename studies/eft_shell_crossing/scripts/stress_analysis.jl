# Experiment 2 — is the post-crossing long-mode departure the kinetic (velocity-dispersion) stress?
#
#   julia --project=. scripts/stress_analysis.jl stress      → results/stress_<name>.h5 (+ printed table)
#
# For each member pair, model (PM, 7LPT), coarse-graining mesh and estimator variant
#   (raw cell variance | gradient-corrected | corrected, multistream cells | corrected, boundary cells):
#   c²_τ(D) = Re⟨Δτ_∥ δ_L*⟩/⟨|δ_L|²⟩,  τ_∥ = k̂_i k̂_j τ̂_ij/ρ̄,  Δ = UV − IR,  δ_L = −i|k|Dψ₁∥   (k ≤ 0.08, binned)
# and the response it implies, integrating the EdS long-mode equation (' = d/dln a)
#   y'' + y'/2 − (3/2) y = c²_τ(a) a,   C2_σ = y/a
# (the stress force −∂_jτ_ij/ρ̄ acting on the long displacement), compared with the measured C2 of the same pairs.
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, HDF5, Printf, Statistics
scratch() = get(ENV, "EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
name = ARGS[1]
cfg, runs = load_config(joinpath(scratch(), name)); pk = planck_pk()
L = cfg["L"]; kf = 2π / L; epochs = Float64.(cfg["epochs"]); E = length(epochs)
meshes = cfg["stress"]["meshes"]; kmax = 0.08
anyr = first(values(runs)); nvec = anyr["nvec"]; kvec = kf .* nvec; km = vec(sqrt.(sum(abs2, kvec; dims=2)))
sel = (km .< cfg["k_split"]) .& (km .<= kmax); kv = kvec[sel, :]; kk = km[sel]; kh = kv ./ kk
comp = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
τpar(m) = sum((a == b ? 1.0 : 2.0) .* kh[:, a] .* kh[:, b] .* m[sel, c] for (c, (a, b)) in enumerate(comp))
longs = sort(unique(k[3] for k in keys(runs))); cache = Dict()
models = ("pm", "lpt")
# c2[model, mesh, variant, epoch], per long seed for the jackknife
num = zeros(length(longs), 2, length(meshes), 4, E); den = zeros(length(longs), E)
scal = zeros(2, length(meshes), 7, E); nsc = zeros(2, E)
for (il, sL) in enumerate(longs), key in keys(runs)
    (key[2] == :UV && key[3] == sL) || continue
    mi = key[1] == "pm" ? 1 : 2
    ir = runs[(key[1], :IR, sL, 0, key[5])]; uv = runs[key]
    ψ1 = get!(() -> linear_psi1_modes(cfg, sL, key[5], nvec, pk; cache=cache), cache, (sL, key[5]))
    x0 = longitudinal(ψ1[sel, :], kv)
    for (e, D) in enumerate(epochs)
        (mi == 1 && D <= cfg["pm"]["D_ini"]) && continue
        δL = -im .* kk .* D .* x0
        mi == 1 && (den[il, e] += sum(abs2, δL))
        for (im_, _) in enumerate(meshes), v in 1:4
            Δ = τpar(ComplexF64.(uv["stress"][e, im_, v, :, :])) .- τpar(ComplexF64.(ir["stress"][e, im_, v, :, :]))
            num[il, mi, im_, v, e] += sum(real.(Δ .* conj.(δL)))
        end
        scal[mi, :, :, e] .+= uv["stress_scalars"][e, :, :]; nsc[mi, e] += 1
    end
end
c2(sel_) = (n = dropdims(sum(num[sel_, :, :, :, :]; dims=1); dims=1); d = vec(sum(den[sel_, :]; dims=1));
            n ./ reshape(d, 1, 1, 1, E))
c2f = c2(1:length(longs))
jk = [c2(setdiff(1:length(longs), [i])) for i in 1:length(longs)]
c2s = sqrt.((length(longs) - 1) / length(longs) .* sum(x -> (x .- sum(jk) ./ length(jk)) .^ 2, jk))
# Green's-function integration in ln a (EdS, D = a), from D_ini with y = y' = 0
function green(c2v)
    lnD = log.(epochs); C = similar(c2v)
    y, yp = 0.0, 0.0; C[1] = 0.0
    for e in 2:E
        h = (lnD[e] - lnD[e-1]) / 200
        for s in 0:199          # RK4 on linear interpolation of the source c²(a)·a
            t = lnD[e-1] + s * h
            f(tt) = (w = (tt - lnD[e-1]) / (lnD[e] - lnD[e-1]); exp(tt) * ((1 - w) * c2v[e-1] + w * c2v[e]))
            rhs(tt, y_, yp_) = (yp_, f(tt) - yp_ / 2 + 1.5 * y_)
            k1 = rhs(t, y, yp); k2 = rhs(t + h/2, y + h/2 * k1[1], yp + h/2 * k1[2])
            k3 = rhs(t + h/2, y + h/2 * k2[1], yp + h/2 * k2[2]); k4 = rhs(t + h, y + h * k3[1], yp + h * k3[2])
            y += h / 6 * (k1[1] + 2k2[1] + 2k3[1] + k4[1]); yp += h / 6 * (k1[2] + 2k2[2] + 2k3[2] + k4[2])
        end
        C[e] = y / epochs[e]
    end
    return C
end
pred = similar(c2f)
for m in 1:2, im_ in eachindex(meshes), v in 1:4
    pred[m, im_, v, :] .= green(replace(c2f[m, im_, v, :], NaN => 0.0))
end
# measured displacement C2 of the same pairs (k ≤ 0.08), PM and 7LPT
edges = collect(0.5:1:(cfg["k_split"] / kf)) .* kf
groups, meta = analyse(cfg, runs, pk; edges=edges, models=["lpt7", "pm"])
G = sum(values(groups)); Rq = G.num[1, :, :, :] ./ G.den[1, :, :, :]
k2 = G.k2x ./ G.den[1, 1, :, :]; k4 = G.k4x ./ G.den[1, 1, :, :]
C2meas = [fit_counterterm(k2[e, :], k4[e, :], Rq[m, e, :], 1 ./ sqrt.(meta.nmodes); kmax=kmax).C2 for m in 1:2, e in 1:E]  # m: 1 = 7LPT, 2 = PM
h5open(joinpath(@__DIR__, "..", "results", "stress_$name.h5"), "w") do h
    h["c2_tau"] = c2f; h["c2_tau_sigma"] = c2s; h["C2_from_stress"] = pred; h["C2_measured_lpt7_pm"] = C2meas
    h["scalars"] = scal ./ reshape(max.(nsc, 1), 2, 1, 1, E); h["epochs"] = epochs; h["meshes"] = Int.(meshes)
    attrs(h)["layout"] = "c2_tau[model(pm,lpt7), mesh, variant(raw,corrected,multistream,boundary), epoch]; scalars[model, mesh, (mass_frac, vol_frac, sigma2_ms, bnd_vol_frac, mean_trace, mean_trace_raw, Nm), epoch]"
end
println("$name: stress response c²_τ [(Mpc/h)²] and the implied C2_σ vs measured ΔC2 = C2_PM − C2_7LPT")
for (im_, Nm) in enumerate(meshes)
    println(" mesh $Nm³:   D   | c²_τ PM (corr)  7LPT (corr) | C2_σ PM−7LPT: corrected   multistream   boundary   raw | measured ΔC2")
    for e in 2:E
        d(v) = pred[1, im_, v, e] - pred[2, im_, v, e]
        @printf "          %.3f | %+9.4f ± %.4f  %+9.4f | %+9.4f  %+9.4f  %+9.4f  %+9.4f | %+9.4f\n" epochs[e] c2f[1, im_, 2, e] c2s[1, im_, 2, e] c2f[2, im_, 2, e] d(2) d(3) d(4) d(1) (C2meas[2, e] - C2meas[1, e])
    end
end
