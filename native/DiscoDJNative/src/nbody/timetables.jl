# ── Timetable lookups used by the N-body steppers (port of DISCO-DJ Cosmology methods) ──
#
# DISCO-DJ evaluates every time-dependent quantity by `jnp.interp` on its timetable
# (linear interpolation, clamped to the end values outside the table).  `_jinterp`
# reproduces that exactly so stepper coefficients match the JAX code to round-off.

export growth_Dplusda, growth_D2plusda, superconft, a_of_superconft, a_of_Dplus, Fplus

"""    _jinterp(x, xp, fp)  — `jnp.interp` semantics (xp ascending; clamp outside)."""
function _jinterp(x::Real, xp::AbstractVector, fp::AbstractVector)
    x <= xp[1] && return fp[1]
    x >= xp[end] && return fp[end]
    i = searchsortedlast(xp, x)                 # xp[i] <= x < xp[i+1]
    i >= length(xp) && return fp[end]
    t = (x - xp[i]) / (xp[i+1] - xp[i])
    return fp[i] + t * (fp[i+1] - fp[i])
end

_need_tt(c) = (isempty(c._superconft_table) || isnan(c._D1_unnormed_at_1)) &&
    error("N-body timetables missing: build the cosmology with compute_timetables (DISCO-DJ defaults)")

_Dplus_tab(c)    = c._D1_table
_Dplusda_tab(c)  = c._f1_table .* c._D1_table ./ c._a_table     # DISCO-DJ: f1·D1/a (normalised)
_D2plusda_tab(c) = c._f2_table .* c._D2_table ./ c._a_table
_D3daplus_tabs(c) = (c._f3a_table .* c._D3a_table ./ c._a_table,       # D3plusada
                     c._f3b_table .* c._D3b_table ./ c._a_table,       # D3plusbda
                     c._D2_table .* _Dplusda_tab(c) .- c._D1_table .* _D2plusda_tab(c))   # D3pluscda

"""    growth_Dplusda(cosmo, a) -> dD₊/da (normalised, D₊(1) = 1)"""
growth_Dplusda(c::Cosmology, a) = (_need_tt(c); _jinterp(a, c._a_table, _Dplusda_tab(c)))
"""    growth_D2plusda(cosmo, a) -> dD₂/da (normalised by D₊,unnormed(1)²)"""
growth_D2plusda(c::Cosmology, a) = (_need_tt(c); _jinterp(a, c._a_table, _D2plusda_tab(c)))
"""    superconft(cosmo, a) -> superconformal time (0 at a = 1)"""
superconft(c::Cosmology, a) = (_need_tt(c); _jinterp(a, c._a_table, c._superconft_table))
"""    a_of_superconft(cosmo, s) -> a"""
a_of_superconft(c::Cosmology, s) = (_need_tt(c); _jinterp(s, c._superconft_table, c._a_table))
"""    a_of_Dplus(cosmo, D) -> a  (inverse of the normalised growth factor)"""
a_of_Dplus(c::Cosmology, D) = (_need_tt(c); _jinterp(D, c._D1_table, c._a_table))
"""    Fplus(cosmo, a) -> a³ E(a) dD₊/da  (momentum growth factor)"""
Fplus(c::Cosmology, a) = a^3 * hubble_E(c, a) * growth_Dplusda(c, a)

# jnp.interp-exact versions of D₊ and D₂ (growth_D1/growth_D2 clamp via Interpolations.jl;
# identical inside the table, kept separate so the steppers follow DISCO-DJ literally)
_Dplus(c::Cosmology, a)  = _jinterp(a, c._a_table, c._D1_table)
_D2plus(c::Cosmology, a) = _jinterp(a, c._a_table, c._D2_table)
