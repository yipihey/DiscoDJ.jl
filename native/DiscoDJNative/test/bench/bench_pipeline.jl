"""
Full nLPT pipeline benchmark (the FFT-dominated path), as opposed to the
single-kernel micro-benchmark in bench_kernels.jl.

Times `compute_lpt` for n_order ∈ {1,2,3} across resolutions and precisions,
for both CPU backends (`:threads`, `:ka`), with a warmup pass and a pure-FFT
reference so the FFT-vs-elementwise split is visible.

Run with:
    JULIA_NUM_THREADS=16 julia --project=native/DiscoDJNative \
        native/DiscoDJNative/test/bench/bench_pipeline.jl [res1 res2 ...]
"""

using DiscoDJNative
using FFTW
using Printf

# resolutions from CLI args, else a default sweep
const RESOLUTIONS = isempty(ARGS) ? [128, 256] : parse.(Int, ARGS)
const BOXSIZE = 1000.0
const SEED    = 42

println("Julia threads : ", Threads.nthreads())
println("FFTW threads  : ", FFTW.get_num_threads())
println("resolutions   : ", RESOLUTIONS)
println()

"Minimum wall time (s) and bytes allocated over `nrep` runs, after one warmup."
function timeit(f; nrep::Int=3)
    f()                                   # warmup (compile + plan)
    best = Inf; bytes = 0
    for _ in 1:nrep
        local t = @timed f()
        best = min(best, t.time)
        bytes = t.bytes
    end
    return best, bytes
end

# Pure-FFT reference: one fwd+inv rfft round trip on this grid/precision.
function fft_roundtrip_time(grid, ::Type{T}) where T
    x = rand(T, grid.res, grid.res, grid.res)
    f = () -> (grid.plan_inv * (grid.plan_fwd * x))
    t, _ = timeit(f)
    return t
end

const c  = Cosmology("Planck18EEBAOSN")
const pk = linear_power_spectrum(c)

@printf("%-5s %-4s %-8s %-8s %10s %10s %10s %8s\n",
        "res", "T", "order", "backend", "time[ms]", "Mcell/s", "alloc[MB]", "fft[ms]")
println("-"^70)

for res in RESOLUTIONS
    ncell = res^3
    for T in (Float32, Float64)
        grid = get_fourier_grid(res, BOXSIZE; T=T)
        fphi = generate_grf(:ngenic, 3, pk, res, BOXSIZE, SEED;
                            dtype=T, dtype_c=Complex{T})
        tfft = fft_roundtrip_time(grid, T) * 1e3
        for n_order in (1, 2, 3)
            for backend in (:threads, :ka)
                t, bytes = timeit(() -> compute_lpt(fphi, grid;
                                                    n_order=n_order, backend=backend))
                @printf("%-5d %-4s %-8d %-8s %10.2f %10.1f %10.1f %8.2f\n",
                        res, T, n_order, backend, t*1e3, ncell/t/1e6,
                        bytes/2^20, tfft)
            end
        end
    end
    println()
end
