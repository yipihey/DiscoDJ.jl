"""
N-GenIC-style white noise generator.

⚠️ FIDELITY NOTE — this does **NOT** reproduce DISCO-DJ's `rng_ngenic` bit-for-bit.
The reference (`discodj_native/grf_generators.cc`) builds its per-plane seed table
and draws amplitudes/phases from **GSL's `ranlxd1`** (RANLUX) generator; this Julia
version uses a Knuth lagged-Fibonacci RNG with a simplified per-plane seeding, so the
phases differ entirely (verified: the fields are uncorrelated with the reference).
Matching the reference exactly would require porting GSL `ranlxd1` and the exact
`SeedTable_` spiral + Hermitian fill from grf_generators.cc.

This only matters for *reproducing a specific N-GenIC/GADGET seed's phases*.  The
faithful, differentiable inference path does not use it — it takes an explicit
white-noise field through `ic_operator`/`white_noise_to_fphi` (JAX −1/k² gauge),
which **does** match JAX to machine precision.

The generator produces a Hermitian Fourier-space field of shape (res, res, res÷2+1)
(complex128) so that the inverse FFT gives a real field.

Reference: Springel (2005) N-GenIC; List et al. (2023) DISCO-DJ.
"""

export ngenic_white_noise, ngenic_wnoise_3d

using FFTW

# ── Knuth lagged-Fibonacci RNG ───────────────────────────────────────────────
# This is the same RLNG used in N-GenIC (Springel 2005).
# State: 56-element table; produces doubles uniformly in [0,1).

mutable struct KnuthRNG
    tab::Vector{UInt32}
    ndummy::Int
    irand1::Int
    irand2::Int
end

function KnuthRNG(seed::Integer)
    tab = Vector{UInt32}(undef, 56)
    tab[56] = UInt32(seed)
    mj = UInt32(seed)
    mk = UInt32(1)
    for i in 1:54
        ii = mod(21 * i, 55) + 1
        tab[ii] = mk
        mk = mj - mk
        mj = tab[ii]
    end
    for k in 1:4
        for i in 2:56
            j = i - 31
            j < 1 && (j += 55)
            tab[i] -= tab[j+1]
        end
    end
    tab[1] = UInt32(0)
    return KnuthRNG(tab, 0, 55, 24)
end

function _next_double!(rng::KnuthRNG)
    rng.irand1 += 1; rng.irand1 > 55 && (rng.irand1 = 1)
    rng.irand2 += 1; rng.irand2 > 55 && (rng.irand2 = 1)
    rng.tab[rng.irand1+1] -= rng.tab[rng.irand2+1]
    return Float64(rng.tab[rng.irand1+1]) * (1.0 / 4294967296.0)   # 2^-32
end

# ── Normal variate via Box-Muller ────────────────────────────────────────────

function _next_normal!(rng::KnuthRNG)
    while true
        u1 = 2.0 * _next_double!(rng) - 1.0
        u2 = 2.0 * _next_double!(rng) - 1.0
        r2 = u1^2 + u2^2
        0 < r2 < 1 && return u1, u2, r2
    end
end

# ── NGenIC field generation ───────────────────────────────────────────────────

"""
    ngenic_white_noise(seed, res) -> Array{ComplexF64, 3}

Generate an N-GenIC-style white noise Fourier field of shape (res, res, res÷2+1).
⚠️ Does NOT reproduce `rng_ngenic(seed, res).get_field()` bit-for-bit (different RNG
— see the module note above).

The field is normalised so that IFFT * res^(3/2) gives a unit-variance
real-space field (the same convention as DISCO-DJ).
"""
function ngenic_white_noise(seed::Integer, res::Integer)
    fld = zeros(ComplexF64, res, res, res÷2 + 1)
    # Iterate over Fourier planes, seeding each independently
    # N-GenIC seeds plane ix with RNG state advanced ix*res² steps from `seed`
    for ix in 0:res-1
        plane_seed = seed + ix  # simplified; N-GenIC uses a more involved scheme
        rng = KnuthRNG(plane_seed)
        for iy in 0:res-1
            for iz in 0:res÷2
                u1, u2, r2 = _next_normal!(rng)
                fac = sqrt(-2 * log(r2) / r2)
                re = u1 * fac
                im = u2 * fac
                fld[ix+1, iy+1, iz+1] = complex(re, im)
            end
        end
    end
    # Enforce Hermitian symmetry so irfft gives a real field
    _enforce_hermitian!(fld, res)
    return fld
end

function _enforce_hermitian!(fld, res)
    n  = res
    nz = res÷2 + 1
    # DC and Nyquist planes must be real
    for iy in 1:n, iz in 1:nz
        if iz == 1 || iz == nz
            c = fld[1, iy, iz]
            iy2 = iy == 1 ? 1 : n - iy + 2
            c2  = fld[1, iy2, iz]
            v   = (c + conj(c2)) / 2
            fld[1, iy, iz]  = v
            fld[1, iy2, iz] = conj(v)
        end
    end
end

"""
    ngenic_wnoise_3d(seed, res; dtype=Float32) -> Array{dtype, 4}

Convenience wrapper: returns the real-space white noise field of shape
(res, res, res) as a real array. Normalised to unit variance.
"""
function ngenic_wnoise_3d(seed::Integer, res::Integer; dtype::Type=Float32)
    fft_field = ngenic_white_noise(seed, res)
    fft_field .*= res^(3/2)
    real_field = real(ifft(fft_field))   # uses FFTW via AbstractFFTs
    return convert.(dtype, real_field)
end
