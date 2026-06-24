"""
Linear growth factors, ported faithfully from DISCO-DJ's
`compute_unnormed_growth` (cosmology.py): a coupled ODE in ln(a) with
log-amplitude state variables, solved by RK4 over the (geometric) a-grid.

Solves for the growing modes
  D1   (Dplus),
  D2   (D2plus,   EdS limit -3/7 a²),
  D3a  (D3plusa,  EdS limit +1/3 a³),
  D3b  (D3plusb,  EdS limit -10/21 a³),
  D3c  (D3plusc,  transverse, EdS limit +1/7 a³),
and the linear growth rate f = d ln D1 / d ln a.  Each Dₙ is normalised by the
unnormalised D1(a=1)ⁿ.  These are the exact-growth factors that multiply the LPT
shape fields ψ₁, ψ₂, ψ₃ₐ, ψ₃ᵦ, ψ₃ᵧ.
"""

export growth_D1, growth_D2, growth_D3a, growth_D3b, growth_D3c, growth_f1, growth_rate

using Interpolations: LinearInterpolation

# ── ODE right-hand side (independent variable t = ln a) ───────────────────────
# State y = [y1,f1, y2,f2, y3a,f3a, y3b,f3b, y3c],  yᵢ = ln|Dᵢ|,  fᵢ = d ln Dᵢ/d ln a.
function _growth_derivs(c::Cosmology{T}, lna, y) where {T}
    a    = exp(lna)
    Esq  = hubble_E(c, a)^2
    Om_a = (Omega_m(c) * a^(-3)) / Esq
    Ox_a = Omega_de(c, a) / Esq
    w_a  = c.w0 + c.wa * (1 - a)
    drag = 1 - T(0.5) * (Om_a + (1 + 3 * w_a) * Ox_a)

    y1, f1, y2, f2, y3a, f3a, y3b, f3b, y3c = y
    D1 = exp(y1); D1sq = D1 * D1; D1cu = D1sq * D1
    D2  = -exp(y2)      # sign(-3/7) = -1
    D3a =  exp(y3a)     # +1
    D3b = -exp(y3b)     # -1
    D3c =  exp(y3c)     # +1

    df1  = T(1.5)*Om_a - f1^2 - f1*drag
    df2  = T(1.5)*Om_a*(1 - D1sq/D2) - f2^2 - f2*drag
    df3a = T(1.5)*Om_a*(1 + 2*D1cu/D3a) - f3a^2 - f3a*drag
    df3b = T(1.5)*Om_a*(1 + (2*D1*D2 - 2*D1cu)/D3b) - f3b^2 - f3b*drag
    dy3c = D1*D2/D3c*(f1 - f2)
    return (f1, df1, f2, df2, f3a, df3a, f3b, df3b, dy3c)
end

@inline _axpy9(y, k, h) = ntuple(i -> y[i] + h*k[i], 9)

function _rk4_growth(c::Cosmology, lna, y, dlna)
    k1 = _growth_derivs(c, lna, y)
    k2 = _growth_derivs(c, lna + dlna/2, _axpy9(y, k1, dlna/2))
    k3 = _growth_derivs(c, lna + dlna/2, _axpy9(y, k2, dlna/2))
    k4 = _growth_derivs(c, lna + dlna,   _axpy9(y, k3, dlna))
    return ntuple(i -> y[i] + dlna/6*(k1[i] + 2k2[i] + 2k3[i] + k4[i]), 9)
end

# ── Build growth tables (called from compute_timetables) ──────────────────────
function _compute_growth_tables(c::Cosmology{T}, a_table::Vector{T}) where {T}
    n = length(a_table)
    amin = a_table[1]
    # EdS initial conditions deep in matter domination.
    y = (log(amin),            one(T),       # D1 ~ a,        f1 = 1
         log(T(3//7)*amin^2),  T(2),         # D2 ~ -3/7 a²,  f2 = 2
         log(T(1//3)*amin^3),  T(3),         # D3a ~ +1/3 a³
         log(T(10//21)*amin^3), T(3),        # D3b ~ -10/21 a³
         log(T(1//7)*amin^3))                # D3c ~ +1/7 a³
    lna = log.(a_table)
    D1=Vector{T}(undef,n); D2=similar(D1); D3a=similar(D1); D3b=similar(D1); D3c=similar(D1); F1=similar(D1)
    setrow!(i, yy) = (D1[i]=exp(yy[1]); F1[i]=yy[2]; D2[i]=-exp(yy[3]);
                      D3a[i]=exp(yy[5]); D3b[i]=-exp(yy[7]); D3c[i]=exp(yy[9]))
    setrow!(1, y)
    for i in 2:n
        y = _rk4_growth(c, lna[i-1], y, lna[i] - lna[i-1])
        setrow!(i, y)
    end
    # Normalise by the unnormalised D1(a=1)^order.
    D1_at1 = LinearInterpolation(a_table, D1)(one(T))
    D1  ./= D1_at1
    D2  ./= D1_at1^2
    D3a ./= D1_at1^3
    D3b ./= D1_at1^3
    D3c ./= D1_at1^3
    return D1, D2, D3a, D3b, D3c, F1
end

# ── Public evaluation functions ───────────────────────────────────────────────
function _geval(c::Cosmology, tbl::Vector, a)
    isempty(c._a_table) && error("Call compute_timetables first")
    LinearInterpolation(c._a_table, tbl)(clamp(a, c._a_table[1], c._a_table[end]))
end

"""    growth_D1(cosmo, a) -> Dplus(a)  (normalised so D₁(1)=1)"""
growth_D1(c::Cosmology, a) = _geval(c, c._D1_table, a)
"""    growth_D2(cosmo, a) -> D2plus(a)  (~ -3/7 D₁²; multiplies ψ₂)"""
growth_D2(c::Cosmology, a) = _geval(c, c._D2_table, a)
"""    growth_D3a(cosmo, a) -> D3plusa(a)  (~ +1/3 D₁³; multiplies ψ₃ₐ)"""
growth_D3a(c::Cosmology, a) = _geval(c, c._D3a_table, a)
"""    growth_D3b(cosmo, a) -> D3plusb(a)  (~ -10/21 D₁³; multiplies ψ₃ᵦ)"""
growth_D3b(c::Cosmology, a) = _geval(c, c._D3b_table, a)
"""    growth_D3c(cosmo, a) -> D3plusc(a)  (transverse, ~ +1/7 D₁³; multiplies ψ₃ᵧ)"""
growth_D3c(c::Cosmology, a) = _geval(c, c._D3c_table, a)
"""    growth_f1(cosmo, a) -> f₁(a) = d ln D₁ / d ln a"""
growth_f1(c::Cosmology, a) = _geval(c, c._f1_table, a)

"""    growth_rate(cosmo, a) -> (D1, D2, f1) at scale factor a"""
growth_rate(c::Cosmology, a) = (growth_D1(c, a), growth_D2(c, a), growth_f1(c, a))
