# ── Ensemble analysis of a configuration's job files ────────────────────────────────────────
#
# Reads the compact per-job HDF5 files written by scripts/run_jobs.jl and reduces them to binned
# statistics, accumulated per *long seed* (the independent unit; jackknife errors are delete-one over
# long seeds).  For every model (lpt1..lptN, pm) and epoch:
#
#   response of the longitudinal displacement   R_ψ(k) = Σ Re Δψ_∥ x_∥* / Σ |x_∥|²,  x = D ψ₁
#   response of the Eulerian density            R_δ(k) = Σ Re Δδ δ_L* / Σ |δ_L|²,     δ_L = −i|k| x_∥
#   response of the Lagrangian velocity dΨ/dD   R_v(k) = Σ Re Δv_∥ ψ₁∥* / Σ |ψ₁∥|²
#   odd-in-L response  (Δ(+L,S) − Δ(−L,S))/2 — the L-independent short×short noise cancels exactly
#   stochastic powers: conditional  Var_{S|L}(Δ) (scatter over short seeds at fixed long modes),
#                      even-in-L    (Δ(+L,S) + Δ(−L,S))/2,  transverse Δψ_⊥,  total |Δ|²
#   density power spectra of both members (T(k) = P_UV/P_IR − 1) and shell-crossed fractions.

using HDF5, TOML

export load_config, analyse, linear_psi1_modes

"""Exact linear displacement shape ψ₁ = −i k φ at D = 1 on the integer wave-vectors `nvec` (coefficients
f_k = rfft/N³), from the IR potential (long modes only) of (seed_L, sign)."""
function linear_psi1_modes(cfg, seed_L, sign, nvec, pk; cache=Dict())
    s = PairSpec(N=cfg["N_ic"], N_ic=cfg["N_ic"], L=cfg["L"], k_split=cfg["k_split"], k_uv=cfg["k_uv"],
                 long_sign=sign, seed_L=seed_L)
    φ = make_fphi(s, pk, :IR; T=Float64, cache=cache)
    N = cfg["N_ic"]; L = cfg["L"]
    out = zeros(ComplexF64, size(nvec, 1), 3)
    for m in axes(nvec, 1)
        nx, ny, nz = nvec[m, :]
        v = φ[mod(nx, N) + 1, mod(ny, N) + 1, nz + 1] / N^3
        for d in 1:3
            out[m, d] = -im * (2π / L) * nvec[m, d] * v
        end
    end
    return out
end

"""All job files of a config directory → Dict((model, member, seedL, seedS, sign) => Dict of arrays)."""
function load_config(dir)
    cfg = TOML.parsefile(joinpath(dir, "config.toml"))
    runs = Dict{Tuple,Dict{String,Any}}()
    for f in readdir(dir)
        m = match(r"^(lpt|pm)_(UV|IR)_L(\d+)_S(\d+)_s([+-]\d)\.h5$", f)
        m === nothing && continue
        key = (m[1], Symbol(m[2]), parse(Int, m[3]), parse(Int, m[4]), parse(Int, m[5]))
        runs[key] = h5open(joinpath(dir, f)) do h
            Dict{String,Any}(k => read(h[k]) for k in keys(h))
        end
        runs[key]["seconds"] = h5open(h -> read_attribute(h, "seconds"), joinpath(dir, f))
    end
    return cfg, runs
end

# displacement / velocity / density modes of (model, member, run) at epoch index e (host ComplexF64)
function _fields(run, model::String, e::Int, D::Float64)
    if startswith(model, "lpt")
        n = parse(Int, model[4:end])
        ψ = run["psi"]                                   # (n_max, M, 3)
        disp = sum(D^m .* ComplexF64.(ψ[m, :, :]) for m in 1:n)
        vel = sum(m * D^(m - 1) .* ComplexF64.(ψ[m, :, :]) for m in 1:n)
        dens = ComplexF64.(run["dens"][e, n, :])
        return disp, vel, dens, run["inv"][e, n]
    else
        return ComplexF64.(run["psi"][e, :, :]), ComplexF64.(run["vel"][e, :, :]),
               ComplexF64.(run["dens"][e, :]), run["inv"][e]
    end
end

struct Acc
    num::Array{Float64,4}     # (quantity, model, epoch, bin)  Σ Re Δ x*
    den::Array{Float64,4}     # (quantity, model, epoch, bin)   Σ |x|² over the runs that model actually has
    pw::Array{Float64,4}      # (powerkind, model, epoch, bin)  Σ |·|²
    cnt::Array{Float64,4}     # (powerkind, model, epoch, bin)  mode counts
    inv::Array{Float64,4}     # (member UV/IR, model, epoch, 2) Σ f, count
    k2x::Matrix{Float64}      # (epoch, bin) Σ k²|x|²  (for |x|²-weighted ⟨k²⟩, ⟨k⁴⟩ per bin)
    k4x::Matrix{Float64}
end
const QUANT = ("disp", "dens", "vel", "disp_odd", "dens_odd")
const PKIND = ("dpsi_par", "dpsi_perp", "ddelta", "cond_psi", "cond_delta", "even_psi", "even_delta",
               "P_uv", "P_ir")
Acc(nm, ne, nb) = Acc(zeros(length(QUANT), nm, ne, nb), zeros(length(QUANT), nm, ne, nb),
                      zeros(length(PKIND), nm, ne, nb), zeros(length(PKIND), nm, ne, nb), zeros(2, nm, ne, 2),
                      zeros(ne, nb), zeros(ne, nb))
Base.:+(a::Acc, b::Acc) = Acc(a.num .+ b.num, a.den .+ b.den, a.pw .+ b.pw, a.cnt .+ b.cnt, a.inv .+ b.inv,
                              a.k2x .+ b.k2x, a.k4x .+ b.k4x)

_q(s) = findfirst(==(s), QUANT); _p(s) = findfirst(==(s), PKIND)

"""
    analyse(cfg, runs, pk; edges, models, mode_mask=nothing) -> (groups::Dict{Int,Acc}, meta)

Per-long-seed accumulators for every model/epoch/bin.  Only modes with |k| < k_split enter (and, if given,
those whose wave-vector passes `mode_mask(k⃗)`, e.g. an isotropy split)."""
function analyse(cfg, runs, pk; edges, models, mode_mask=nothing)
    epochs = Float64.(cfg["epochs"]); ne = length(epochs); nb = length(edges) - 1; nm = length(models)
    anyrun = first(values(runs)); nvec = anyrun["nvec"]
    L = cfg["L"]; kvec = (2π / L) .* nvec; kmag = vec(sqrt.(sum(abs2, kvec; dims=2)))
    sel = kmag .< cfg["k_split"]
    mode_mask === nothing || (sel .&= [mode_mask(kvec[m, :]) for m in axes(kvec, 1)])
    kvec = kvec[sel, :]; kmag = kmag[sel]; bi = kbin_index(kmag, edges)
    longs = sort(unique(k[3] for k in keys(runs)))
    cache = Dict()
    groups = Dict{Int,Acc}()
    for sL in longs
        A = Acc(nm, ne, nb)
        signs = sort(unique(k[5] for k in keys(runs) if k[3] == sL))
        shorts = sort(unique(k[4] for k in keys(runs) if k[3] == sL && k[2] == :UV))
        ψ1 = Dict(sg => linear_psi1_modes(cfg, sL, sg, nvec, pk; cache=cache)[sel, :] for sg in signs)
        for (im_, model) in enumerate(models), (e, D) in enumerate(epochs)
            mtype = startswith(model, "lpt") ? "lpt" : "pm"
            (mtype == "pm" && D <= get(get(cfg, "pm", Dict()), "D_ini", 0.0)) && continue
            Δ = Dict(); P = Dict()
            for sg in signs
                ir = get(runs, (mtype, :IR, sL, 0, sg), nothing); ir === nothing && continue
                dI, vI, δI, fI = _fields(ir, model, e, D)
                dI = dI[sel, :]; vI = vI[sel, :]; δI = δI[sel]
                A.inv[2, im_, e, 1] += fI; A.inv[2, im_, e, 2] += 1
                for sS in shorts
                    uv = get(runs, (mtype, :UV, sL, sS, sg), nothing); uv === nothing && continue
                    dU, vU, δU, fU = _fields(uv, model, e, D)
                    A.inv[1, im_, e, 1] += fU; A.inv[1, im_, e, 2] += 1
                    Δ[(sg, sS)] = (dU[sel, :] .- dI, vU[sel, :] .- vI, δU[sel] .- δI)
                    P[(sg, sS)] = (δU[sel], δI)
                end
            end
            isempty(Δ) && continue
            for ((sg, sS), (dψ, dv, dδ)) in Δ
                x = D .* longitudinal(ψ1[sg], kvec); xδ = -im .* kmag .* x; xv = x ./ D
                dpar = longitudinal(dψ, kvec); dperp = transverse(dψ, kvec); vpar = longitudinal(dv, kvec)
                for m in eachindex(bi)
                    b = bi[m]; b == 0 && continue
                    A.num[_q("disp"), im_, e, b] += real(dpar[m] * conj(x[m]))
                    A.num[_q("dens"), im_, e, b] += real(dδ[m] * conj(xδ[m]))
                    A.num[_q("vel"), im_, e, b] += real(vpar[m] * conj(xv[m]))
                    A.den[_q("disp"), im_, e, b] += abs2(x[m]); A.den[_q("dens"), im_, e, b] += abs2(xδ[m])
                    A.den[_q("vel"), im_, e, b] += abs2(xv[m])
                    if im_ == 1
                        A.k2x[e, b] += kmag[m]^2 * abs2(x[m]); A.k4x[e, b] += kmag[m]^4 * abs2(x[m])
                    end
                    for (k_, v) in (("dpsi_par", abs2(dpar[m])), ("dpsi_perp", sum(abs2, dperp[m, :])),
                                    ("ddelta", abs2(dδ[m])), ("P_uv", abs2(P[(sg, sS)][1][m])),
                                    ("P_ir", abs2(P[(sg, sS)][2][m])))
                        A.pw[_p(k_), im_, e, b] += v; A.cnt[_p(k_), im_, e, b] += 1
                    end
                end
            end
            # odd / even in the long modes (pairs (+L,S), (−L,S)); x of the +L member
            if 1 in signs && -1 in signs
                x = D .* longitudinal(ψ1[1], kvec); xδ = -im .* kmag .* x
                for sS in shorts
                    (haskey(Δ, (1, sS)) && haskey(Δ, (-1, sS))) || continue
                    op = (Δ[(1, sS)][1] .- Δ[(-1, sS)][1]) ./ 2; ep = (Δ[(1, sS)][1] .+ Δ[(-1, sS)][1]) ./ 2
                    oδ = (Δ[(1, sS)][3] .- Δ[(-1, sS)][3]) ./ 2; eδ = (Δ[(1, sS)][3] .+ Δ[(-1, sS)][3]) ./ 2
                    opar = longitudinal(op, kvec); epar = longitudinal(ep, kvec)
                    for m in eachindex(bi)
                        b = bi[m]; b == 0 && continue
                        A.num[_q("disp_odd"), im_, e, b] += real(opar[m] * conj(x[m]))
                        A.num[_q("dens_odd"), im_, e, b] += real(oδ[m] * conj(xδ[m]))
                        A.den[_q("disp_odd"), im_, e, b] += abs2(x[m]); A.den[_q("dens_odd"), im_, e, b] += abs2(xδ[m])
                        A.pw[_p("even_psi"), im_, e, b] += abs2(epar[m]); A.cnt[_p("even_psi"), im_, e, b] += 1
                        A.pw[_p("even_delta"), im_, e, b] += abs2(eδ[m]); A.cnt[_p("even_delta"), im_, e, b] += 1
                    end
                end
            end
            # conditional scatter over short seeds at fixed (L, sign): Var_{S|L}, unbiased
            for sg in signs
                ks_ = [sS for sS in shorts if haskey(Δ, (sg, sS))]
                ns = length(ks_); ns < 2 && continue
                mψ = sum(longitudinal(Δ[(sg, s)][1], kvec) for s in ks_) ./ ns
                mδ = sum(Δ[(sg, s)][3] for s in ks_) ./ ns
                for s in ks_
                    rψ = longitudinal(Δ[(sg, s)][1], kvec) .- mψ; rδ = Δ[(sg, s)][3] .- mδ
                    for m in eachindex(bi)
                        b = bi[m]; b == 0 && continue
                        A.pw[_p("cond_psi"), im_, e, b] += abs2(rψ[m]) * ns / (ns - 1)
                        A.pw[_p("cond_delta"), im_, e, b] += abs2(rδ[m]) * ns / (ns - 1)
                        A.cnt[_p("cond_psi"), im_, e, b] += 1; A.cnt[_p("cond_delta"), im_, e, b] += 1
                    end
                end
            end
        end
        groups[sL] = A
    end
    kb = [mean(kmag[bi .== b]) for b in 1:nb]; nmodes = [count(==(b), bi) for b in 1:nb]
    return groups, (k=kb, nmodes=nmodes, edges=edges, epochs=epochs, models=models, longs=longs)
end
