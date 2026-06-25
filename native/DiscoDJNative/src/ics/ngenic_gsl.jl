"""
Bit-exact N-GenIC white noise — reproduces DISCO-DJ's `rng_ngenic`
(`discodj_native/grf_generators.cc`) so a specific N-GenIC/GADGET seed's phases can
be reproduced.

This ports, faithfully:
1. GSL's `ranlxd1` generator (RANLUX double, luxury = 202) — `gsl_rng_set` /
   `gsl_rng_uniform` (validated bit-for-bit against the GSL library);
2. the `rng_ngenic` seed-table spiral and Hermitian field fill.

`ngenic_field_gsl(seed, res)` returns the Fourier white-noise field
(res, res, res÷2+1) identical to `rng_ngenic(seed, res).get_field()`.
`ngenic_wnoise_real(seed, res)` returns the real-space white noise identical to
DISCO-DJ's `get_ngenic_wnoise` (= irfftn(field · res^(3/2))).  Feed the latter to
`white_noise_to_fphi` (the faithful −1/k² IC map) to reproduce a DISCO-DJ NGenIC
simulation's initial conditions exactly.
"""

export ngenic_field_gsl, ngenic_wnoise_real, ngenic_ic

# ── GSL ranlxd1 (RANLUX double, luxury = 202) ─────────────────────────────────
const _RANLXD_ONE_BIT = 1.0 / 281474976710656.0   # 1/2^48
@inline _ranlxd_next(i::Int) = (i + 1) % 12        # GSL `next[]` for 0-based index

mutable struct RanlxdState
    xd::Vector{Float64}
    carry::Float64
    ir::Int
    jr::Int
    ir_old::Int
    pr::Int
end

"""    ranlxd1_set(seed) -> RanlxdState   (GSL `gsl_rng_set` for ranlxd1)"""
function ranlxd1_set(s::Integer)
    s == 0 && (s = 1)                               # GSL default seed
    i = UInt64(s) & 0xFFFFFFFF
    xbit = zeros(Int, 31)
    for k in 1:31
        xbit[k] = Int(i & 1); i >>= 1
    end
    ibit = 0; jbit = 18
    xd = zeros(Float64, 12)
    for k in 1:12
        x = 0.0
        for _ in 1:48
            y = Float64((xbit[ibit+1] + 1) % 2)
            x += x + y
            xbit[ibit+1] = (xbit[ibit+1] + xbit[jbit+1]) % 2
            ibit = (ibit + 1) % 31; jbit = (jbit + 1) % 31
        end
        xd[k] = _RANLXD_ONE_BIT * x
    end
    return RanlxdState(xd, 0.0, 11, 7, 0, 202)
end

# GSL `increment_state` (xd indexed 1-based; logical 0-based indices i → xd[i+1]).
function _ranlxd_increment!(st::RanlxdState)
    xd = st.xd; carry = st.carry; ir = st.ir; jr = st.jr; k = 0
    ob = _RANLXD_ONE_BIT
    while ir > 0
        y1 = xd[jr+1] - xd[ir+1]; y2 = y1 - carry
        if y2 < 0; carry = ob; y2 += 1; else; carry = 0.0; end
        xd[ir+1] = y2; ir = _ranlxd_next(ir); jr = _ranlxd_next(jr); k += 1
    end
    kmax = st.pr - 12
    while k <= kmax                                 # unrolled 12-word RANLUX_STEPs
        y1 = xd[8] - xd[1]; y1 -= carry
        y2 = xd[9]-xd[2];  if y1<0; y2-=ob; y1+=1; end; xd[1]=y1
        y3 = xd[10]-xd[3]; if y2<0; y3-=ob; y2+=1; end; xd[2]=y2
        y1 = xd[11]-xd[4]; if y3<0; y1-=ob; y3+=1; end; xd[3]=y3
        y2 = xd[12]-xd[5]; if y1<0; y2-=ob; y1+=1; end; xd[4]=y1
        y3 = xd[1]-xd[6];  if y2<0; y3-=ob; y2+=1; end; xd[5]=y2
        y1 = xd[2]-xd[7];  if y3<0; y1-=ob; y3+=1; end; xd[6]=y3
        y2 = xd[3]-xd[8];  if y1<0; y2-=ob; y1+=1; end; xd[7]=y1
        y3 = xd[4]-xd[9];  if y2<0; y3-=ob; y2+=1; end; xd[8]=y2
        y1 = xd[5]-xd[10]; if y3<0; y1-=ob; y3+=1; end; xd[9]=y3
        y2 = xd[6]-xd[11]; if y1<0; y2-=ob; y1+=1; end; xd[10]=y1
        y3 = xd[7]-xd[12]; if y2<0; y3-=ob; y2+=1; end; xd[11]=y2
        if y3<0; carry=ob; y3+=1; else; carry=0.0; end; xd[12]=y3
        k += 12
    end
    kmax = st.pr
    while k < kmax
        y1 = xd[jr+1]-xd[ir+1]; y2 = y1 - carry
        if y2<0; carry=ob; y2+=1; else; carry=0.0; end
        xd[ir+1]=y2; ir=_ranlxd_next(ir); jr=_ranlxd_next(jr); k += 1
    end
    st.ir=ir; st.ir_old=ir; st.jr=jr; st.carry=carry
    return st
end

"""    gsl_uniform!(st) -> Float64 ∈ [0,1)   (GSL `gsl_rng_uniform` for ranlxd1)"""
function gsl_uniform!(st::RanlxdState)
    ir = st.ir; st.ir = _ranlxd_next(ir)
    st.ir == st.ir_old && _ranlxd_increment!(st)
    return st.xd[st.ir + 1]
end

# ── N-GenIC field (port of rng_ngenic in grf_generators.cc) ───────────────────
# Seed table: the symmetric spiral of `0x7fffffff * uniform`, filled by a master
# ranlxd1 seeded with `seed`.
function _ngenic_seedtable(seed::Integer, nres::Int)
    rng = ranlxd1_set(seed)
    tab = zeros(UInt32, nres * nres)
    draw() = UInt32(floor(2147483647.0 * gsl_uniform!(rng)))   # 0x7fffffff * uniform → uint
    half = nres ÷ 2
    @inbounds for i in 0:half-1
        for j in 0:i-1;  tab[i*nres + j + 1]                     = draw(); end
        for j in 0:i;    tab[j*nres + i + 1]                     = draw(); end
        for j in 0:i-1;  tab[(nres-1-i)*nres + j + 1]            = draw(); end
        for j in 0:i;    tab[(nres-1-j)*nres + i + 1]            = draw(); end
        for j in 0:i-1;  tab[i*nres + (nres-1-j) + 1]            = draw(); end
        for j in 0:i;    tab[j*nres + (nres-1-i) + 1]            = draw(); end
        for j in 0:i-1;  tab[(nres-1-i)*nres + (nres-1-j) + 1]   = draw(); end
        for j in 0:i;    tab[(nres-1-j)*nres + (nres-1-i) + 1]   = draw(); end
    end
    return tab
end

"""
    ngenic_field_gsl(seed, res) -> Array{ComplexF64,3}  (res, res, res÷2+1)

Bit-exact reproduction of `rng_ngenic(seed, res).get_field()`.
"""
function ngenic_field_gsl(seed::Integer, res::Int)
    nresp = res ÷ 2 + 1
    half  = res ÷ 2
    tab   = _ngenic_seedtable(seed, res)
    fld   = zeros(ComplexF64, res, res, nresp)
    twopi = 2 * π
    @inbounds for i in 0:res-1
        ii = i > 0 ? res - i : 0
        for j in 0:res-1
            jj = j > 0 ? res - j : 0
            rng = ranlxd1_set(tab[i*res + j + 1])
            for k in 0:half
                phase = gsl_uniform!(rng) * twopi
                ampl = 0.0
                while true                                   # reject 0 and 1
                    ampl = gsl_uniform!(rng)
                    (ampl == 0.0 || ampl == 1.0) || break
                end
                (i == half || j == half || k == half) && continue   # Nyquist planes
                (i == 0 && j == 0 && k == 0) && continue            # DC
                ampl = sqrt(-log(ampl))
                zrand = complex(ampl * cos(phase), ampl * sin(phase))
                if k > 0
                    fld[i+1, j+1, k+1] = zrand
                else                                          # k=0 plane: Hermitian
                    if i == 0
                        if j < half
                            fld[i+1, j+1, k+1]  = zrand
                            fld[i+1, jj+1, k+1] = conj(zrand)
                        end
                    elseif i < half
                        fld[i+1, j+1, k+1]   = zrand
                        fld[ii+1, jj+1, k+1] = conj(zrand)
                    end
                end
            end
        end
    end
    return fld
end

"""
    ngenic_wnoise_real(seed, res; dtype=Float64) -> Array{dtype,3}  (res,res,res)

Real-space N-GenIC white noise, identical to DISCO-DJ's `get_ngenic_wnoise`
(= irfftn(field · res^(3/2))).  Feed to `white_noise_to_fphi` (the faithful −1/k²
IC map) to reproduce a DISCO-DJ NGenIC simulation's ICs exactly.
"""
function ngenic_wnoise_real(seed::Integer, res::Int; dtype::Type{<:AbstractFloat}=Float64)
    fld = ngenic_field_gsl(seed, res) .* res^(3/2)
    return convert.(dtype, irfft(fld, res, [3, 1, 2]))
end

"""
    ngenic_ic(seed, res, boxsize, pk_table; T=Float64) -> φ(k)

Faithful initial Fourier potential for an N-GenIC `seed`: the bit-exact N-GenIC
white noise put through the faithful (JAX −1/k²) IC map (`ic_operator` /
`white_noise_to_fphi`).  A drop-in `fphi` for `compute_core` / `lpt_displacement`
(the differentiable path) that reproduces a DISCO-DJ NGenIC simulation's ICs.
"""
function ngenic_ic(seed::Integer, res::Int, boxsize::Real, pk_table::Dict;
                   T::Type{<:AbstractFloat}=Float64)
    op = ic_operator(res, boxsize, pk_table; T=T)
    return white_noise_to_fphi(op, ngenic_wnoise_real(seed, res; dtype=T))
end
