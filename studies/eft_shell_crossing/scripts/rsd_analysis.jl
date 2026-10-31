# Redshift-space response (line of sight ẑ, plane-parallel) → results/rsd_<name>.h5
#
#   julia --project=. scripts/rsd_analysis.jl rg_ks20 ...
#
# s = x + u_z ẑ with u = dx/dln a (EdS: f = 1).  For each member pair, model (lpt1..7, pm), epoch and μ bin
# (μ = |k_z|/k):
#   R_s(k, μ) = Σ Re[Δδ_s δ_K*] / Σ |δ_K|²,  δ_K = (1 + f μ²) δ_L   (Kaiser-linear long field)
# fitted per μ bin as −C2(μ) k² (k ≤ 0.08), and C2(μ) = C2₀ + C2_μ μ² across bins.  The realization-level
# residual PM − LPT_n in redshift space, and the real- vs redshift-space shell-crossed fractions, test whether
# the pre-crossed single-stream sheet stays exact through redshift-space crossing.
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, HDF5, Printf, Statistics
scratch() = get(ENV, "EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")

function rsd(name)
    cfg, runs = load_config(joinpath(scratch(), name)); pk = planck_pk()
    L = cfg["L"]; kf = 2π / L; epochs = Float64.(cfg["epochs"]); E = length(epochs)
    nmax = cfg["lpt"]["n_max"]; models = vcat(["lpt$n" for n in 1:nmax], ["pm"]); nm = length(models)
    anyr = first(values(runs)); nvec = anyr["nvec"]; kvec = kf .* nvec; km = vec(sqrt.(sum(abs2, kvec; dims=2)))
    sel = (km .< cfg["k_split"]) .& (km .<= 0.081)
    kk = km[sel]; μ = abs.(kvec[sel, 3]) ./ kk
    μedges = [0.0, 0.4, 0.7, 1.0001]; nμ = 3; μb = [searchsortedlast(μedges, x) for x in μ]
    edges = collect(0.5:1:(0.081 / kf)) .* kf; nb = length(edges) - 1; kb = kbin_index(kk, edges)
    longs = sort(unique(k[3] for k in keys(runs))); cache = Dict()
    num = zeros(length(longs), nm, E, nμ, nb); den = zeros(length(longs), nm, E, nμ, nb); k2s = zeros(nμ, nb)
    invr = zeros(2, nm, E); cnt = zeros(nm, E)
    for (il, sL) in enumerate(longs), key in keys(runs)
        (key[2] == :UV && key[3] == sL) || continue
        ir = runs[(key[1], :IR, sL, 0, key[5])]; uv = runs[key]
        ψ1 = get!(() -> linear_psi1_modes(cfg, sL, key[5], nvec, pk; cache=cache), cache, (sL, key[5]))
        δ1 = -im .* kk .* longitudinal(ψ1[sel, :], kvec[sel, :])
        for (i, m) in enumerate(models)
            (startswith(m, "lpt") != (key[1] == "lpt")) && continue
            n = m == "pm" ? 0 : parse(Int, m[4:end])
            for (e, D) in enumerate(epochs)
                (m == "pm" && D <= cfg["pm"]["D_ini"]) && continue
                du = m == "pm" ? uv["dens_rsd"][e, :] : uv["dens_rsd"][e, n, :]
                di = m == "pm" ? ir["dens_rsd"][e, :] : ir["dens_rsd"][e, n, :]
                Δ = ComplexF64.(du[sel]) .- ComplexF64.(di[sel]); x = D .* (1 .+ μ .^ 2) .* δ1
                for j in eachindex(Δ)
                    b = kb[j]; b == 0 && continue
                    num[il, i, e, μb[j], b] += real(Δ[j] * conj(x[j])); den[il, i, e, μb[j], b] += abs2(x[j])
                    (i == 1 && e == E) && (k2s[μb[j], b] += kk[j]^2)
                end
                ri = m == "pm" ? uv["inv"][e] : uv["inv"][e, n]; rs = m == "pm" ? uv["inv_rsd"][e] : uv["inv_rsd"][e, n]
                invr[1, i, e] += ri; invr[2, i, e] += rs; cnt[i, e] += 1
            end
        end
    end
    nmodes = dropdims(sum(den[1:1, 1:1, end:end, :, :] .> 0; dims=(1, 2, 3)); dims=(1, 2, 3))
    k2b = [sum(kk[(kb .== b) .& (μb .== q)] .^ 2) / max(count((kb .== b) .& (μb .== q)), 1) for q in 1:nμ, b in 1:nb]
    function C2μ(sel_)
        R = dropdims(sum(num[sel_, :, :, :, :]; dims=1) ./ sum(den[sel_, :, :, :, :]; dims=1); dims=1)
        out = fill(NaN, nm, E, nμ)
        for i in 1:nm, e in 1:E, q in 1:nμ
            r = R[i, e, q, :]; ok = isfinite.(r) .& (k2b[q, :] .> 0)
            sum(ok) >= 2 && (out[i, e, q] = -sum(r[ok] .* k2b[q, ok]) / sum(k2b[q, ok] .^ 2))
        end
        return out
    end
    full = C2μ(1:length(longs))
    jk = [C2μ(setdiff(1:length(longs), [l])) for l in 1:length(longs)]
    jkσ(f) = (v = [f(x) for x in jk]; m = sum(v) ./ length(v); sqrt.((length(v) - 1) / length(v) .* sum(x -> (x .- m) .^ 2, v)))
    σ = jkσ(identity)
    ip_ = nm
    σd = [jkσ(x -> x[ip_, :, :] .- x[i, :, :]) for i in 1:nm]          # jackknife error of PM − model i (realization level)
    # C2(μ) = C2₀ + C2_μ μ² across the μ bins (bin-mean μ²)
    μ2 = [mean(μ[μb .== q] .^ 2) for q in 1:nμ]
    lin(c) = (A = hcat(ones(nμ), μ2); A \ c)
    coef = [lin(full[i, e, :]) for i in 1:nm, e in 1:E]
    h5open(joinpath(@__DIR__, "..", "results", "rsd_$name.h5"), "w") do h
        h["C2_mu"] = full; h["C2_mu_sigma"] = σ; h["mu2"] = μ2; h["epochs"] = epochs
        h["resid_sigma"] = cat([reshape(x, 1, size(x)...) for x in σd]...; dims=1)   # (model, epoch, μ) error of PM − model
        h["C2_0"] = [c[1] for c in coef]; h["C2_mu2"] = [c[2] for c in coef]
        h["inv_real"] = invr[1, :, :] ./ max.(cnt, 1); h["inv_rsd"] = invr[2, :, :] ./ max.(cnt, 1)
        attrs(h)["models"] = join(models, ","); attrs(h)["mu_edges"] = μedges
    end
    ip = nm; i7 = nmax
    println("$name redshift space (LOS z): C2(μ) [(Mpc/h)²] for μ bins $(round.(sqrt.(μ2), digits=2)) and inverted fractions (real / s-space), PM")
    for (e, D) in enumerate(epochs)
        D < 0.1 && continue
        @printf "  D=%.3f  PM−5LPT %s | PM−7LPT %s | inv real %.4f  s %.4f\n" D join([@sprintf("%+8.4f±%.4f", full[ip, e, q] - full[5, e, q], σd[5][e, q]) for q in 1:nμ], " ") join([@sprintf("%+8.4f±%.4f", full[ip, e, q] - full[i7, e, q], σd[i7][e, q]) for q in 1:nμ], " ") invr[1, ip, e] / max(cnt[ip, e], 1) invr[2, ip, e] / max(cnt[ip, e], 1)
    end
end

foreach(rsd, ARGS)
