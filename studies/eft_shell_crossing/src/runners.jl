# ── LPT and PM runs of one member of a pair ─────────────────────────────────────────────────
#
# Both runners return compact, host-side results on a common `ModeSet`:
#   LPT:  ψ̂_m (order m = 1..n_max, shape fields at D = 1) — the displacement / velocity of any order
#         n at any D follow exactly as Σ_{m≤n} Dᵐ ψ̂_m and Σ m Dᵐ⁻¹ ψ̂_m (EdS), so they are not stored
#         per epoch; the *nonlinear* functionals (density, inverted fraction) are measured per
#         (epoch, order) on the device.
#   PM:   per epoch ψ̂ (displacement X − q), Π̂ = dΨ/dD, δ̂, inverted fraction, total momentum.

export run_lpt, run_pm, pm_ics, d_grid

"""
    run_lpt(fphi, K, ms, L, epochs; n_max) -> NamedTuple

`fphi`: device initial potential; `K`: device `NLPTKernels`.  Returns
  psi   (n_max, M, 3) ComplexF64 — ψ̂_m at D = 1,
  dens  (E, n_max, M) — CIC δ̂ of the order-n map at each epoch (deconvolved),
  inv   (E, n_max)    — inverted-tetrahedron fraction,
  rms   (E, n_max)    — rms |Ψ| (Mpc/h)."""
function run_lpt(fphi, K, ms::ModeSet, L::Real, epochs::AbstractVector; n_max::Int, mode::Symbol=:fast,
                 ms_band=nothing, cis=nothing, cis_orders=Int[], cis_epochs=Float64[], rsd::Bool=false, fgrowth=D -> 1.0,
                 stress=nothing, stress_order::Int=0)
    sh = compute_core(fphi, K; n_order=n_max, mode=mode)
    ψ1 = sh["psi_1"]; T = eltype(ψ1); N = size(ψ1, 1)
    M = length(ms); E = length(epochs)
    psi = zeros(ComplexF64, n_max, M, 3)
    for m in 1:n_max
        psi[m, :, :] .= vector_modes(ms, sh["psi_$m"])
    end
    band = ms_band === nothing ? nothing :
        cat((reshape(vector_modes(ms_band, sh["psi_$m"]), 1, :, 3) for m in 1:n_max)...; dims=1)
    dens = zeros(ComplexF64, E, n_max, M); invf = zeros(E, n_max); rms = zeros(E, n_max)
    q = lattice_q(ψ1, N, L)
    Ψ = similar(ψ1); X = similar(ψ1); mesh = similar(ψ1, T, N, N, N); cnt = similar(ψ1, Int32, N, N, N)
    co = [n for n in cis_orders if n <= n_max]
    # redshift space (line of sight ẑ): s = x + u_z ẑ, u = dΨ/dln a = f D Π, Π = dΨ/dD = Σ m Dᵐ⁻¹ ψ_m
    dens_s = rsd ? zeros(ComplexF64, E, n_max, M) : nothing; inv_s = rsd ? zeros(E, n_max) : nothing
    Π = (rsd || stress !== nothing) ? similar(ψ1) : nothing; Ψs = rsd ? similar(ψ1) : nothing
    st = stress === nothing ? nothing : zeros(ComplexF64, E, length(stress), 4, M, 6)
    stsc = stress === nothing ? nothing : zeros(E, length(stress), 7)
    ce = [e for (e, D) in enumerate(epochs) if any(isapprox(D, x; atol=1e-9) for x in cis_epochs)]
    cisd = cis === nothing ? nothing : zeros(Float32, length(ce), length(co), length(cis.R), nchan(cis), (N ÷ cis.stride)^3)
    for (e, D) in enumerate(epochs)
        fill!(Ψ, zero(T)); Π === nothing || fill!(Π, zero(T))
        for n in 1:n_max
            Ψ .+= T(D^n) .* sh["psi_$n"]
            Π === nothing || (Π .+= T(n * D^(n - 1)) .* sh["psi_$n"])
            if stress !== nothing && n == stress_order
                U = T(fgrowth(D) * D) .* Π
                for (ims, msm) in enumerate(stress)
                    r = stress_modes(msm, Ψ, U, q, L, msm.N)
                    st[e, ims, :, :, :] .= r.modes
                    stsc[e, ims, :] .= (r.mass_frac, r.vol_frac, r.sigma2_ms, r.bnd_vol_frac, r.mean_trace, r.mean_trace_raw, msm.N)
                end
                U = nothing
            end
            if rsd
                Ψs .= Ψ; Ψs[:, :, :, 3] .+= T(fgrowth(D) * D) .* view(Π, :, :, :, 3)
                dens_s[e, n, :] .= density_modes(ms, Ψs, q; mesh=mesh, X=X)
                inv_s[e, n] = inverted_fraction(Ψs, L; cnt=cnt)
            end
            if cisd !== nothing && n in co && e in ce
                dm, cs = density_modes(ms, Ψ, q; mesh=mesh, X=X, cis=cis)
                dens[e, n, :] .= dm; cisd[findfirst(==(e), ce), findfirst(==(n), co), :, :, :] .= cs
            else
                dens[e, n, :] .= density_modes(ms, Ψ, q; mesh=mesh, X=X)
            end
            invf[e, n] = inverted_fraction(Ψ, L; cnt=cnt)
            rms[e, n] = sqrt(sum(abs2, Ψ) / N^3)
        end
    end
    sh = nothing
    return (psi=psi, dens=dens, inv=invf, rms=rms, band=band, cis=cisd, cis_orders=co, dens_s=dens_s, inv_s=inv_s,
            stress=st, stress_scalars=stsc)
end

"""Time grid uniform in D from D_ini to D_end with ≈`n_steps` steps, every epoch inserted as a node."""
function d_grid(D_ini, D_end, n_steps, epochs)
    g = collect(range(D_ini, D_end, length=n_steps + 1))
    for D in epochs
        (D <= D_ini || D > D_end) && continue
        i = argmin(abs.(g .- D)); g[i] = D
    end
    g = sort(unique(g)); g[1] = D_ini; g[end] = D_end
    return g
end

"""
    pm_ics(cosmo, fphi, K, D_ini; ic_order=3, mode=:fast) -> (Ψ0, Π0)

nLPT initial conditions (EdS growth) for the PM run at D_ini."""
function pm_ics(c, fphi, K, D_ini; ic_order::Int=3, mode::Symbol=:fast)
    sh = compute_core(fphi, K; n_order=ic_order, mode=mode)
    Ψ0, Π0 = nbody_ics_lpt(c, sh, D_ini; n_order=ic_order)
    return Ψ0, Π0
end

"""
    run_pm(cosmo, Ψ0, Π0, ms, L, epochs; D_ini, n_steps, res_pm, stepper=:bullfrog, band=nothing, settings...)

PM evolution (DiscoDJNative `run_nbody`, BullFrog stepper on a grid uniform in D with every epoch a
node) from the initial displacement Ψ0 and Π0 = dΨ/dD at D_ini.  `settings` go to `run_nbody`
(worder, grad_kernel_order, deconvolve, antialias, ...).  Returns per-epoch psi (E,M,3),
vel (E,M,3) [Π = dΨ/dD = P/F₊], dens (E,M), inv (E), rms (E), mom (E,3) [Σ P / (N³ rms|P|)], the D
grid and the wall time.  With `band = (ms_band, epochs_band)` the displacement is also stored on a
second (typically larger) mode set at a few epochs (band-growth diagnostics)."""
function run_pm(c, Ψ0, Π0, ms::ModeSet, L::Real, epochs::AbstractVector; D_ini::Real, n_steps::Int,
                res_pm::Int, stepper::Symbol=:bullfrog, band=nothing, cis=nothing, cis_epochs=Float64[], rsd::Bool=false,
                stress=nothing, settings...)
    T = eltype(Ψ0); N = size(Ψ0, 1); M = length(ms)
    ep = [D for D in epochs if D > D_ini]
    E = length(epochs)
    psi = zeros(ComplexF64, E, M, 3); vel = zeros(ComplexF64, E, M, 3); dens = zeros(ComplexF64, E, M)
    invf = fill(NaN, E); rms = fill(NaN, E); mom = fill(NaN, E, 3)
    bpsi = band === nothing ? nothing : zeros(ComplexF64, length(band[2]), length(band[1]), 3)
    ce = [e for (e, D) in enumerate(epochs) if any(isapprox(D, x; atol=1e-9) for x in cis_epochs)]
    cisd = cis === nothing ? nothing : zeros(Float32, length(ce), length(cis.R), nchan(cis), (N ÷ cis.stride)^3)
    dens_s = rsd ? zeros(ComplexF64, E, M) : nothing; inv_s = rsd ? fill(NaN, E) : nothing
    st = stress === nothing ? nothing : zeros(ComplexF64, E, length(stress), 4, M, 6)
    stsc = stress === nothing ? nothing : fill(NaN, E, length(stress), 7)
    q = lattice_q(Ψ0, N, L)
    mesh = similar(Ψ0, T, N, N, N); Xb = similar(Ψ0); cnt = similar(Ψ0, Int32, N, N, N); Ψ = similar(Ψ0)
    grid = d_grid(D_ini, maximum(epochs), n_steps, vcat(ep, band === nothing ? Float64[] : band[2]))
    snap = Dict(round(D, digits=10) => findfirst(==(D), epochs) for D in ep)
    bsnap = band === nothing ? Dict() : Dict(round(D, digits=10) => i for (i, D) in enumerate(band[2]))
    function cb(k, a, X, P)
        e = get(snap, round(a, digits=10), nothing); b = get(bsnap, round(a, digits=10), nothing)
        (e === nothing && b === nothing) && return
        for d in 1:3
            Ψ[:, :, :, d] .= view(X, :, :, :, d) .- q[d]
        end
        b === nothing || (bpsi[b, :, :] .= vector_modes(band[1], Ψ))
        e === nothing && return
        psi[e, :, :] .= vector_modes(ms, Ψ)
        Π = P ./ T(DJ.Fplus(c, a))
        vel[e, :, :] .= vector_modes(ms, Π)
        if stress !== nothing   # u = dx/dln a = f D Π
            fD = T(DJ.growth_Dplusda(c, a) * a)
            U = fD .* Π
            for (ims, msm) in enumerate(stress)
                r = stress_modes(msm, Ψ, U, q, L, msm.N)
                st[e, ims, :, :, :] .= r.modes
                stsc[e, ims, :] .= (r.mass_frac, r.vol_frac, r.sigma2_ms, r.bnd_vol_frac, r.mean_trace, r.mean_trace_raw, msm.N)
            end
            U = nothing
        end
        if rsd   # s = x + u_z ẑ, u = dΨ/dln a = f D Π  (EdS: f = 1, D = a)
            f = DJ.growth_Dplusda(c, a) * a / DJ._Dplus(c, a)
            Π[:, :, :, 1] .= view(Ψ, :, :, :, 1); Π[:, :, :, 2] .= view(Ψ, :, :, :, 2)
            Π[:, :, :, 3] .= view(Ψ, :, :, :, 3) .+ T(f * DJ._Dplus(c, a)) .* view(Π, :, :, :, 3)
            dens_s[e, :] .= density_modes(ms, Π, q; mesh=mesh, X=Xb)
            inv_s[e] = inverted_fraction(Π, L; cnt=cnt)
        end
        Π = nothing
        if cisd === nothing || !(e in ce)
            dens[e, :] .= density_modes(ms, Ψ, q; mesh=mesh, X=Xb)
        else
            dm, cs = density_modes(ms, Ψ, q; mesh=mesh, X=Xb, cis=cis)
            dens[e, :] .= dm; cisd[findfirst(==(e), ce), :, :, :] .= cs
        end
        invf[e] = inverted_fraction(Ψ, L; cnt=cnt)
        rms[e] = sqrt(sum(abs2, Ψ) / N^3)
        prms = sqrt(sum(abs2, P) / N^3)
        for d in 1:3
            mom[e, d] = sum(view(P, :, :, :, d)) / N^3 / prms
        end
    end
    t = @elapsed run_nbody(c, Ψ0, Π0; boxsize=L, a_ini=D_ini, a_end=grid[end], n_steps=nothing,
                           time_var=grid, res_pm=res_pm, stepper=stepper,
                           return_displacement=true, step_callback=cb,
                           callback_when=(k, a) -> haskey(snap, round(a, digits=10)) || haskey(bsnap, round(a, digits=10)),
                           settings...)
    return (psi=psi, vel=vel, dens=dens, inv=invf, rms=rms, mom=mom, grid=grid, seconds=t, band=bpsi, cis=cisd, dens_s=dens_s, inv_s=inv_s,
            stress=st, stress_scalars=stsc)
end
