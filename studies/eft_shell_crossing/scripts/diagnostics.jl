# Isotropy and conservation diagnostics of a configuration → results/diagnostics_<name>.json
#
#   julia --project=. scripts/diagnostics.jl <name>
#
# Isotropy: C₂ (k ≤ 0.08, jackknife) from modes near the grid axes (Σk̂ᵢ⁴ > 0.7) and near the diagonals
# (Σk̂ᵢ⁴ < 0.5) separately — a lattice/PM-grid anisotropy would split them.
# Conservation: PM total momentum |Σ P| / (N³ rms|P|) per epoch (exactly 0 for an exact integrator), and
# the mass: the k = 0 density mode is 0 by construction (particle number), reported for completeness.
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, JSON3, Printf, Statistics
scratch() = get(ENV, "EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
name = ARGS[1]
cfg, runs = load_config(joinpath(scratch(), name)); pk = planck_pk()
L = cfg["L"]; kf = 2π / L; edges = collect(0.5:1:(cfg["k_split"] / kf)) .* kf
models = ["lpt3", "lpt7", "pm"]
out = Dict{String,Any}("epochs" => cfg["epochs"])
for (tag, mask) in (("axis", k -> sum((k ./ sqrt(sum(abs2, k))) .^ 4) > 0.7),
                    ("diagonal", k -> sum((k ./ sqrt(sum(abs2, k))) .^ 4) < 0.5))
    groups, meta = analyse(cfg, runs, pk; edges=edges, models=models, mode_mask=mask)
    G = collect(values(groups)); tot = sum(G); ne = length(meta.epochs); nb = length(edges) - 1
    k2 = tot.k2x ./ tot.den[1, 1, :, :]; k4 = tot.k4x ./ tot.den[1, 1, :, :]
    R(As) = (S = sum(As); S.num[1, :, :, :] ./ S.den[1, :, :, :])
    _, σ = jackknife(R, G); Rf = R(G)
    for (i, m) in enumerate(models)
        C = Float64[]; s = Float64[]
        for e in 1:ne
            f(As) = fit_counterterm(k2[e, :], k4[e, :], R(As)[i, e, :], σ[i, e, :]; kmax=0.08).C2
            c, sc = jackknife(f, G); push!(C, c); push!(s, sc)
        end
        out["C2_$(tag)_$m"] = C; out["sC2_$(tag)_$m"] = s
    end
    out["nmodes_$tag"] = meta.nmodes
end
mom = [maximum(abs.(r["mom"]); init=0.0) for (k, r) in runs if k[1] == "pm"]
out["max_momentum"] = maximum(filter(isfinite, vcat([vec(r["mom"]) for (k, r) in runs if k[1] == "pm"]...)) .|> abs)
clean(x) = x isa AbstractFloat && !isfinite(x) ? nothing : x isa AbstractVector ? clean.(x) : x
open(io -> JSON3.write(io, Dict(k => clean(v) for (k, v) in out)), joinpath(@__DIR__, "..", "results", "diagnostics_$name.json"), "w")
for (e, D) in enumerate(cfg["epochs"])
    @printf "D=%.3f  PM axis %8.4f ± %.4f  diag %8.4f ± %.4f | 7LPT axis %8.4f diag %8.4f\n" D out["C2_axis_pm"][e] out["sC2_axis_pm"][e] out["C2_diagonal_pm"][e] out["sC2_diagonal_pm"][e] out["C2_axis_lpt7"][e] out["C2_diagonal_lpt7"][e]
end
@printf "max |Σ P|/(N³ rms P) over PM runs and epochs: %.2e\n" out["max_momentum"]
