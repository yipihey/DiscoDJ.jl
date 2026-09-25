# ── N-body time integration: port of DISCO-DJ `run_nbody` (disco_dj.py) and its steppers ──
#
#   steppers   :bullfrog, :fastpm  (DKDPiIntegrator: momentum variable Π = dΨ/dD, D-time drifts)
#              :symplectic         (DKDSymplecticIntegrator: superconformal-time drifts)
#   time_var   :a, :log_a, :D, :superconft, or an explicit vector of scale factors
#   ICs        arrays (Ψ, Π = dΨ/dD) — `nbody_ics_lpt` builds them from DiscoDJNative's nLPT shapes
#              exactly as DISCO-DJ `evaluate_lpt_psi_at_a` / `D_derivative=True`; `bullfrog_ics`
#              is the ic_method="bullfrog" path (one discreteness-suppressed BullFrog step 0 → a_ini)
#   output     positions (wrapped) or displacements, canonical momentum P = a² dx/dt, final a
#              (or every step with `collect_all`), optional per-step `step_callback`.
#
# The state is (Ψ, Mom) with Ψ the *unwrapped* displacement; one step is drift–kick–drift:
#   Ψ ← Ψ + ddrift1·Mom;  Mom ← α·Mom + β·acc(Ψ);  Ψ ← Ψ + ddrift2·Mom
# with acc from `pm_acceleration` and the coefficients of `stepper_coefficients`.
#
# Not ported (not needed for method="pm" forward runs): nufftpm / treepm / 1-D exact forces,
# Diffrax, and the custom-VJP adjoint scan (DiscoDJNative has its own AD strategy).

export stepper_coefficients, run_nbody, nbody_ics_lpt, bullfrog_ics

# numpy.linspace (step = Δ/num; y = arange·step + start; y[end] = stop)
_nplinspace(s, e, num::Int) = (st = (e - s) / (num - 1); y = [s + i * st for i in 0:num-1]; y[end] = e; y)

const _GL5_X = (-0.906179845938664, -0.5384693101056831, 0.0, 0.5384693101056831, 0.906179845938664)
const _GL5_W = (0.23692688505618928, 0.4786286704993663, 0.5688888888888887, 0.4786286704993663, 0.23692688505618928)

function _a_to_internal(c, tv::Symbol, a)
    tv === :a && return a
    tv === :log_a && return log(a)
    tv === :D && return _Dplus(c, a)
    tv === :superconft && return superconft(c, a)
    error("time_var must be :a, :log_a, :D, :superconft or a vector of scale factors")
end
function _internal_to_a(c, tv::Symbol, x)
    tv === :a && return x
    tv === :log_a && return exp(x)
    tv === :D && return a_of_Dplus(c, x)
    tv === :superconft && return a_of_superconft(c, x)
    error("unknown time_var $tv")
end

"""
    stepper_coefficients(cosmo, stepper, a_ini, a_end, n_steps; time_var=:D)
        -> (; alpha, beta, ddrift1, ddrift2, all_a)

Per-step drift/kick coefficients of the DISCO-DJ steppers (`get_integrator_args`), on the time grid
of `all_a_and_internal` (uniform in the internal time variable, midpoints taken in that variable)."""
function stepper_coefficients(c::Cosmology, stepper::Symbol, a_ini::Real, a_end::Real, n_steps::Int;
                              time_var=:D)
    _need_tt(c)
    if time_var isa AbstractVector
        length(time_var) == n_steps + 1 || error("time_var vector must have n_steps + 1 entries")
        tgrid = _nplinspace(0.0, Float64(n_steps), n_steps + 1)
        itoa = x -> _jinterp(x, tgrid, Float64.(time_var))
        all_int = _nplinspace(0.0, Float64(n_steps), n_steps + 1)
    else
        s, e = _a_to_internal(c, time_var, Float64(a_ini)), _a_to_internal(c, time_var, Float64(a_end))
        itoa = x -> _internal_to_a(c, time_var, x)
        all_int = _nplinspace(s, e, n_steps + 1)
    end
    all_a = itoa.(all_int)
    int_mid = (all_int[1:end-1] .+ all_int[2:end]) ./ 2
    a_mid = itoa.(int_mid)
    ab, ae = all_a[1:end-1], all_a[2:end]
    if stepper === :bullfrog
        D1 = c._D1_unnormed_at_1
        Du(a) = _Dplus(c, a) * D1;               Ddau(a) = growth_Dplusda(c, a) * D1
        D2u(a) = _D2plus(c, a) * D1^2;           D2dau(a) = growth_D2plusda(c, a) * D1^2
        all_D = Du.(all_a); all_Dda = Ddau.(all_a); all_D2 = D2u.(all_a); all_D2da = D2dau.(all_a)
        D_b, D_e = all_D[1:end-1], all_D[2:end]
        dD = D_e .- D_b
        xi = D_b ./ dD
        Dda_b, Dda_e = all_Dda[1:end-1], all_Dda[2:end]
        D2_b = all_D2[1:end-1]
        D2da_b, D2da_e = all_D2da[1:end-1], all_D2da[2:end]
        bracket = (D2_b .+ D2da_b ./ Dda_b .* dD ./ 2) ./ ((xi .+ 0.5) .* dD) .- (xi .+ 0.5) .* dD
        alpha = (D2da_e ./ Dda_e .- bracket) ./ (D2da_b ./ Dda_b .- bracket)
        D_mid = _Dplus.(Ref(c), a_mid)                   # normalised
        beta = (1 .- alpha) ./ D_mid
        dd1 = D_mid .- D_b ./ D1
        dd2 = D_e ./ D1 .- D_mid
    elseif stepper === :fastpm
        alpha = Fplus.(Ref(c), ab) ./ Fplus.(Ref(c), ae)
        D_mid = _Dplus.(Ref(c), a_mid)
        beta = (1 .- alpha) ./ D_mid
        all_D = _Dplus.(Ref(c), all_a)
        dd1 = D_mid .- all_D[1:end-1]
        dd2 = all_D[2:end] .- D_mid
    elseif stepper === :symplectic
        alpha = ones(length(a_mid))
        f(a) = 1 / (a^2 * hubble_E(c, a))
        integ(b, e) = 0.5 * (e - b) * sum(_GL5_W[i] * f(0.5 * (e - b) * _GL5_X[i] + 0.5 * (b + e)) for i in 1:5)
        beta = 1.5 * Omega_m(c) .* integ.(ab, ae)
        sc = superconft.(Ref(c), all_a); scm = superconft.(Ref(c), a_mid)
        dd1 = scm .- sc[1:end-1]
        dd2 = sc[2:end] .- scm
    else
        error("stepper must be :bullfrog, :fastpm or :symplectic")
    end
    return (alpha=alpha, beta=beta, ddrift1=dd1, ddrift2=dd2, all_a=all_a)
end

# momentum variable conversions (Π = dΨ/dD ↔ integrator momentum)
_pi_to_v(c, stepper, Π, a) = stepper === :symplectic ? Π .* Fplus(c, a) : Π
_v_to_pi(c, stepper, v, a) = stepper === :symplectic ? v ./ Fplus(c, a) : v

"""
    run_nbody(cosmo, psi_ini, pi_ini; boxsize, a_ini, a_end, n_steps, res_pm,
              time_var=:D, stepper=:bullfrog, antialias=0, grad_kernel_order=4,
              laplace_kernel_order=0, worder=2, deconvolve=false, n_resample=1,
              resampling_method=:fourier, collect_all=false, return_all_a=false,
              return_displacement=false, step_callback=nothing)
        -> (X_or_Ψ, P, a_out)

Port of DISCO-DJ `run_nbody(method="pm")` from given initial displacement `psi_ini::(n,n,n,3)`
and `pi_ini = dΨ/dD` at `a_ini` (defaults as in DISCO-DJ).  Returns positions wrapped to
[0, L) (or the displacement with `return_displacement`), the canonical momentum P = a² dx/dt,
and `a_end` (or every step's a with `collect_all`/`return_all_a`; with `collect_all` the first
two outputs are vectors over steps 0..n_steps).  `step_callback(k, a_k, X, P)` is called for
k = 0..n_steps with unwrapped X = q + Ψ.  Runs on the backend of `psi_ini` (CPU / CUDA)."""
function run_nbody(c::Cosmology, psi_ini::AbstractArray{T,4}, pi_ini::AbstractArray{T,4};
                   boxsize::Real, a_ini::Real, a_end::Real, n_steps::Union{Int,Nothing}, res_pm::Int,
                   time_var=:D, stepper::Symbol=:bullfrog, antialias::Int=0, grad_kernel_order::Int=4,
                   laplace_kernel_order::Int=0, worder::Int=2, deconvolve::Bool=false,
                   n_resample::Int=1, resampling_method::Symbol=:fourier,
                   collect_all::Bool=false, return_all_a::Bool=false, return_displacement::Bool=false,
                   step_callback=nothing) where {T}
    n = size(psi_ini, 1); L = Float64(boxsize)
    if n_steps === nothing
        time_var isa AbstractVector || error("n_steps = nothing requires an explicit time_var vector")
        n_steps = length(time_var) - 1
    end
    co = stepper_coefficients(c, stepper, a_ini, a_end, n_steps; time_var=time_var)
    cfg = PMConfig(res_pm=res_pm, n_part=n, boxsize=L, worder=worder, deconvolve=deconvolve,
                   antialias=antialias, grad_order=grad_kernel_order, lap_order=laplace_kernel_order,
                   n_resample=n_resample, resampling=resampling_method)
    K = PMKernels(cfg, T, psi_ini)
    acc(x) = pm_acceleration(x, cfg; kernels=K)

    Psi = copy(psi_ini)
    Mom = T.(_pi_to_v(c, stepper, pi_ini, a_ini))
    q1 = _lagr1d(n, L, T)
    dev(x) = (y = similar(psi_ini, eltype(x), size(x)); copyto!(y, x); y)
    qd = (dev(reshape(q1, n, 1, 1)), dev(reshape(q1, 1, n, 1)), dev(reshape(q1, 1, 1, n)))
    addq(ψ) = (X = similar(ψ); for d in 1:3; X[:, :, :, d] .= view(ψ, :, :, :, d) .+ qd[d]; end; X)
    canon(M, a) = T.(_v_to_pi(c, stepper, M, a) .* Fplus(c, a))

    step_callback === nothing || step_callback(0, co.all_a[1], addq(Psi), canon(Mom, co.all_a[1]))
    hist_psi = collect_all ? [copy(Psi)] : nothing
    hist_mom = collect_all ? [copy(Mom)] : nothing
    for k in 1:n_steps
        Psi .+= T(co.ddrift1[k]) .* Mom
        Mom .= T(co.alpha[k]) .* Mom .+ T(co.beta[k]) .* acc(Psi)
        Psi .+= T(co.ddrift2[k]) .* Mom
        step_callback === nothing || step_callback(k, co.all_a[k+1], addq(Psi), canon(Mom, co.all_a[k+1]))
        if collect_all
            push!(hist_psi, copy(Psi)); push!(hist_mom, copy(Mom))
        end
    end
    wrap(ψ) = (X = addq(ψ); X .= mod.(X .+ T(L), T(L)); X)
    if collect_all
        P = [canon(hist_mom[i], co.all_a[i]) for i in eachindex(hist_mom)]
        Xo = [return_displacement ? hist_psi[i] : wrap(hist_psi[i]) for i in eachindex(hist_psi)]
        return Xo, P, co.all_a
    end
    P = canon(Mom, a_end)
    return (return_displacement ? Psi : wrap(Psi)), P, (return_all_a ? co.all_a : a_end)
end

"""
    nbody_ics_lpt(cosmo, shapes, a; n_order, exact_growth=false) -> (Ψ, Π = dΨ/dD)

DISCO-DJ `evaluate_lpt_psi_at_a` and `_evaluate_lpt_property_at_a(D_derivative=true)` from the
nLPT shape fields of `compute_core` (EdS: Ψ = Σ Dⁿψₙ, Π = Σ n Dⁿ⁻¹ψₙ) or `compute_core_exact`
(exact growth up to 3rd order, Π from the dDₖ/da tables divided by dD₊/da)."""
function nbody_ics_lpt(c::Cosmology, shapes::AbstractDict, a::Real; n_order::Int, exact_growth::Bool=false)
    D = _Dplus(c, a)
    ψ1 = shapes["psi_1"]; T = eltype(ψ1)
    Ψ = zero(ψ1); Π = zero(ψ1)
    if exact_growth
        n_order <= 3 || error("exact growth is available up to 3rd order")
        tf = 1 / growth_Dplusda(c, a)
        Ψ .+= T(D) .* ψ1;                         Π .+= T(tf * growth_Dplusda(c, a)) .* ψ1
        if n_order >= 2
            Ψ .+= T(_D2plus(c, a)) .* shapes["psi_2_ex"]
            Π .+= T(tf * growth_D2plusda(c, a)) .* shapes["psi_2_ex"]
        end
        if n_order == 3
            da, db, dc = _D3daplus_tabs(c)
            for (key, tab, dtab) in (("psi_3a_ex", c._D3a_table, da), ("psi_3b_ex", c._D3b_table, db),
                                     ("psi_3c_ex", c._D3c_table, dc))
                Ψ .+= T(_jinterp(a, c._a_table, tab)) .* shapes[key]
                Π .+= T(tf * _jinterp(a, c._a_table, dtab)) .* shapes[key]
            end
        end
    else
        for n in 1:n_order
            ψn = shapes["psi_$n"]
            Ψ .+= T(D^n) .* ψn
            Π .+= T(n * D^(n - 1)) .* ψn
        end
    end
    return Ψ, Π
end

"""
    bullfrog_ics(cosmo, psi1, a_ini; boxsize, res_pm, settings...) -> (Ψ, Π)

DISCO-DJ `ic_method="bullfrog"`: one BullFrog step from a = 0 to `a_ini` starting from 1LPT
(Ψ = D(0)ψ₁, Π = ψ₁), with the discreteness-suppressing defaults of arXiv:2309.10865
(antialias=1, ik gradient, −1/k², PCS, deconvolution, 2³ Fourier sheet resampling); any of
them can be overridden through `settings`."""
function bullfrog_ics(c::Cosmology, psi1::AbstractArray{T,4}, a_ini::Real; boxsize::Real, res_pm::Int,
                      settings...) where {T}
    s = merge((time_var=:D, stepper=:bullfrog, antialias=1, grad_kernel_order=0, laplace_kernel_order=0,
               worder=4, deconvolve=true, n_resample=2, resampling_method=:fourier), NamedTuple(settings))
    D0 = _Dplus(c, 0.0)
    Ψ, P, _ = run_nbody(c, T(D0) .* psi1, copy(psi1); boxsize=boxsize, a_ini=0.0, a_end=a_ini, n_steps=1,
                        res_pm=res_pm, return_displacement=true, s...)
    return Ψ, P ./ T(Fplus(c, a_ini))
end
