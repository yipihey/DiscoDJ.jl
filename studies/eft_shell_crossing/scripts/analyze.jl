# Reduce a configuration's job files to compact binned statistics (results/<name>.h5).
#
#   julia --project=. scripts/analyze.jl <name> [<name> ...]      (reads $EFTSC_SCRATCH/<name>/)
#
# Errors: delete-one jackknife over long seeds when ≥ 3 long seeds are available; otherwise a
# mode-counting estimate σ_R² = ⟨|Δ − R x|²⟩ / (2 Σ|x|²) per bin (single-realization tests).
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, HDF5, Printf, Statistics
scratch() = get(ENV, "EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")

const KMAXS = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18]

function reduce_config(name)
    dir = joinpath(scratch(), name)
    cfg, runs = load_config(dir)
    pk = planck_pk()
    L = cfg["L"]; kf = 2π / L
    edges = collect(0.5:1:(cfg["k_split"] / kf)) .* kf
    nlpt = haskey(cfg, "lpt") && any(k -> k[1] == "lpt", keys(runs)) ? cfg["lpt"]["n_max"] : 0
    models = vcat(["lpt$n" for n in 1:nlpt], any(k -> k[1] == "pm", keys(runs)) ? ["pm"] : String[])
    t = @elapsed groups, meta = analyse(cfg, runs, pk; edges=edges, models=models)
    println("$name: $(length(groups)) long seeds, $(length(runs)) runs, models $(models); analysed in $(round(t, digits=1)) s")
    G = collect(values(groups)); tot = sum(G)
    nm, ne, nb = length(models), length(meta.epochs), length(edges) - 1
    k2 = tot.k2x ./ tot.den[EFTShell._q("disp"), 1, :, :]; k4 = tot.k4x ./ tot.den[EFTShell._q("disp"), 1, :, :]
    out = joinpath(@__DIR__, "..", "results", "$name.h5"); mkpath(dirname(out))
    h5open(out, "w") do h
        h["k"] = meta.k; h["nmodes"] = meta.nmodes; h["edges"] = edges; h["epochs"] = meta.epochs
        h["k2"] = k2; h["k4"] = k4
        attrs(h)["models"] = join(models, ","); attrs(h)["n_long"] = length(G)
        attrs(h)["n_runs"] = length(runs); attrs(h)["config"] = read(joinpath(dir, "config.toml"), String)
        attrs(h)["run_seconds"] = sum(r["seconds"] for r in values(runs))
        # shell-crossed fractions
        h["inv_uv"] = tot.inv[1, :, :, 1] ./ tot.inv[1, :, :, 2]
        h["inv_ir"] = tot.inv[2, :, :, 1] ./ tot.inv[2, :, :, 2]
        for (qi, q) in enumerate(EFTShell.QUANT)
            all(iszero, tot.den[qi, :, :, :]) && continue
            R(As) = (S = sum(As); S.num[qi, :, :, :] ./ S.den[qi, :, :, :])
            Rf = R(G)
            if length(G) >= 3
                _, σ = jackknife(R, G)
            else   # mode counting: residual variance per bin
                pk_ = q in ("disp", "disp_odd") ? "dpsi_par" : q in ("dens", "dens_odd") ? "ddelta" : nothing
                σ = fill(NaN, nm, ne, nb)
                if pk_ !== nothing
                    pi_ = EFTShell._p(pk_)
                    for i in 1:nm, e in 1:ne, b in 1:nb
                        n = tot.cnt[pi_, i, e, b]; n > 0 || continue
                        var = max(tot.pw[pi_, i, e, b] - Rf[i, e, b] * tot.num[qi, i, e, b], 0) / n
                        σ[i, e, b] = sqrt(var / (2 * tot.den[qi, i, e, b] / n) / n)
                    end
                end
            end
            g = create_group(h, q); g["R"] = Rf; g["sigma"] = σ
            # counterterm fits: C2 (order 1) and C2, C4 (order 2) vs k_max; jackknife over long seeds
            for (order, tag) in ((1, "fit1"), (2, "fit2"))
                C2 = fill(NaN, nm, ne, length(KMAXS)); C4 = similar(C2); s2 = similar(C2); s4 = similar(C2); chi = similar(C2)
                for i in 1:nm, e in 1:ne, (j, km) in enumerate(KMAXS)
                    w = σ[i, e, :]
                    f = fit_counterterm(k2[e, :], k4[e, :], Rf[i, e, :], w; kmax=km, order=order)
                    C2[i, e, j] = f.C2; C4[i, e, j] = f.C4; chi[i, e, j] = f.chi2 / max(f.dof, 1)
                    if length(G) >= 3
                        fj(As) = (Rj = R(As)[i, e, :]; ff = fit_counterterm(k2[e, :], k4[e, :], Rj, w; kmax=km, order=order); [ff.C2, ff.C4])
                        _, sj = jackknife(fj, G); s2[i, e, j] = sj[1]; s4[i, e, j] = sj[2]
                    else
                        s2[i, e, j] = f.σC2; s4[i, e, j] = f.σC4
                    end
                end
                gg = create_group(g, tag); gg["C2"] = C2; gg["sC2"] = s2; gg["C4"] = C4; gg["sC4"] = s4; gg["chi2dof"] = chi
                gg["kmax"] = KMAXS
            end
        end
        # PM − LPTn residual response (same realizations: the common short×short noise cancels)
        if "pm" in models && length(G) >= 3
            ip = findfirst(==("pm"), models)
            for q in ("disp", "disp_odd", "dens", "vel")
                qi = EFTShell._q(q); all(iszero, tot.den[qi, :, :, :]) && continue
                Rq(As) = (S = sum(As); S.num[qi, :, :, :] ./ S.den[qi, :, :, :])
                g = create_group(h, "resid_" * q)
                C2 = fill(NaN, nm, ne, length(KMAXS)); s2 = similar(C2)
                dR = fill(NaN, nm, ne, nb); sdR = similar(dR)
                for i in 1:nm
                    i == ip && continue
                    d(As) = (R = Rq(As); R[ip, :, :] .- R[i, :, :])
                    v, s = jackknife(d, G); dR[i, :, :] = v; sdR[i, :, :] = s
                    for e in 1:ne, (j, km) in enumerate(KMAXS)
                        fj(As) = fit_counterterm(k2[e, :], k4[e, :], d(As)[e, :], s[e, :]; kmax=km).C2
                        c, sc = jackknife(fj, G); C2[i, e, j] = c; s2[i, e, j] = sc
                    end
                end
                g["dR"] = dR; g["sigma"] = sdR; g["C2"] = C2; g["sC2"] = s2; g["kmax"] = KMAXS
            end
        end
        for (pi_, p) in enumerate(EFTShell.PKIND)
            all(iszero, tot.cnt[pi_, :, :, :]) && continue
            P(As) = (S = sum(As); L^3 .* S.pw[pi_, :, :, :] ./ S.cnt[pi_, :, :, :])
            g = create_group(h, "P_" * p); g["P"] = P(G)
            length(G) >= 3 && (g["sigma"] = jackknife(P, G)[2])
        end
        # linear spectra of the regressors (per bin, per epoch): ψ_L∥ = D ψ₁∥ and δ_L = −i|k| ψ_L∥
        nper = tot.cnt[EFTShell._p("dpsi_par"), 1, :, :]
        h["P_lin_psi"] = L^3 .* tot.den[EFTShell._q("disp"), 1, :, :] ./ nper
        h["P_lin_delta"] = L^3 .* tot.den[EFTShell._q("dens"), 1, :, :] ./ nper
        # T(k) = P_UV/P_IR − 1 (density, ratio of ensemble sums)
        Tk(As) = (S = sum(As); S.pw[EFTShell._p("P_uv"), :, :, :] ./ S.pw[EFTShell._p("P_ir"), :, :, :] .- 1)
        h["T"] = Tk(G); length(G) >= 3 && (h["T_sigma"] = jackknife(Tk, G)[2])
    end
    println("wrote $out")
end

foreach(reduce_config, ARGS)
