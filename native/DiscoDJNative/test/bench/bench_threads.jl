# Minimal thread-scaling probe: time 2LPT at res=256 fp32 for the current thread
# count.  Run at several JULIA_NUM_THREADS to find the FFT sweet spot (the nLPT
# pipeline is FFT-bound, and FFTW scales with threads up to ~one socket).
#   for n in 8 16 32 64; do JULIA_NUM_THREADS=$n julia --project=native/DiscoDJNative \
#       native/DiscoDJNative/test/bench/bench_threads.jl; done
using DiscoDJNative, FFTW, Printf
const res = isempty(ARGS) ? 256 : parse(Int, ARGS[1])
const T = Float32
c    = Cosmology("Planck18EEBAOSN")
pk   = linear_power_spectrum(c)
grid = get_fourier_grid(res, 1000.0; T=T)
fphi = generate_grf(:ngenic, 3, pk, res, 1000.0, 42; dtype=T, dtype_c=Complex{T})
compute_lpt(fphi, grid; n_order=2, backend=:ka)          # warmup
best = Inf
for _ in 1:3
    global best = min(best, @elapsed compute_lpt(fphi, grid; n_order=2, backend=:ka))
end
@printf("threads=%-3d (fftw=%-3d)  2LPT res=%d fp32 : %7.1f ms  (%.1f Mcell/s)\n",
        Threads.nthreads(), FFTW.get_num_threads(), res, best*1e3, res^3/best/1e6)
