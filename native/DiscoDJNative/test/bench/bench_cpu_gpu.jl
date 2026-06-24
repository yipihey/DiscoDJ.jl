# CPU vs GPU nLPT benchmark — the headline crossover result.
#
# Times the full `compute_lpt` pipeline on the CPU (Threads backend) and the GPU
# (CUDA backend, via the DiscoDJNativeCUDAExt extension) across resolutions, in
# Float32, and reports wall-clock, Mcell/s, and speedup.  The shared GRF is
# canonicalised once so both backends run identical inputs.
#
# Needs an environment with CUDA available alongside DiscoDJNative.  Suggested:
#   JULIA_CUDA_HARD_MEMORY_LIMIT=38GiB JULIA_NUM_THREADS=32 \
#     julia --project=<env-with-DiscoDJNative+CUDA> \
#       native/DiscoDJNative/test/bench/bench_cpu_gpu.jl 128 256 512
using DiscoDJNative, CUDA, Printf
@assert CUDA.functional() "CUDA not functional"

const RES   = isempty(ARGS) ? [128, 256] : parse.(Int, ARGS)
const ORDER = 2
const L     = 1000.0
const T     = Float32

best(f; n=3) = (f(); minimum(@elapsed(f()) for _ in 1:n))            # warmup + min
gbest(f; n=3) = (CUDA.@sync(f()); minimum(@elapsed(CUDA.@sync f()) for _ in 1:n))

println("Julia/FFTW threads=$(Threads.nthreads())")
println("GPU: ", CUDA.name(CUDA.device()), "   precision=$T   nLPT order=$ORDER")
@printf("%-5s %12s %10s %12s %10s %9s\n",
        "res", "CPU[ms]", "CPU Mc/s", "GPU[ms]", "GPU Mc/s", "speedup")
println("-"^66)

c = Cosmology("Planck18EEBAOSN"); pk = linear_power_spectrum(c)
for res in RES
    ncell = res^3
    grid = get_fourier_grid(res, L; T=T)
    fphi = generate_grf(:ngenic, 3, pk, res, L, 42; dtype=T, dtype_c=Complex{T})
    fc   = canonicalize_hermitian(fphi, grid)

    tcpu = best(() -> compute_lpt(fc, grid; n_order=ORDER, backend=:threads))

    gridg = to_gpu(grid); fphig = to_gpu(fphi)
    tgpu = gbest(() -> compute_lpt(fphig, gridg; n_order=ORDER, backend=:ka))
    CUDA.unsafe_free!(fphig)                       # release device memory between sizes
    GC.gc(); CUDA.reclaim()

    @printf("%-5d %12.2f %10.1f %12.2f %10.1f %8.1fx\n",
            res, tcpu*1e3, ncell/tcpu/1e6, tgpu*1e3, ncell/tgpu/1e6, tcpu/tgpu)
end
