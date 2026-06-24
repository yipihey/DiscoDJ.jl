"""
KA-CPU vs Threads.@threads benchmark suite.

Hypothesis from GraphGP.jl: KA on CPU adds dispatch overhead vs raw @threads.
Tests each kernel in DiscoDJNative across resolutions res ∈ [64, 128, 256, 512].

Run with:
    julia --project=native/DiscoDJNative --threads=auto \
          native/DiscoDJNative/test/bench/bench_kernels.jl
"""

using BenchmarkTools
using KernelAbstractions
using DiscoDJNative

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 2.0
BenchmarkTools.DEFAULT_PARAMETERS.samples = 50

println("Julia threads: ", Threads.nthreads())
println()

RESOLUTIONS = [64, 128, 256]  # add 512 for the full benchmark (slow first-run JIT)

println("=" ^ 70)
println("Kernel benchmarks: KA-CPU vs Threads.@threads")
println("=" ^ 70)
println()

for res in RESOLUTIONS
    nk = res * res * (res÷2 + 1)
    println("res = $res  (n_modes = $nk)")
    println("-" ^ 50)

    k2   = rand(Float64, res, res, res÷2+1)
    k2[1,1,1] = 0.0
    kx   = rand(Float64, res, res, res÷2+1)
    f    = randn(ComplexF64, res, res, res÷2+1)
    f2   = randn(ComplexF64, res, res, res÷2+1)
    out  = similar(f)
    out2 = similar(f)

    # ── inv_laplace ────────────────────────────────────────────────────────────
    b_ka  = @benchmark inv_laplace_ka!($out, $f, $k2)  setup=(out .= 0) evals=1
    b_thr = @benchmark inv_laplace_threads!($out, $f, $k2) setup=(out .= 0) evals=1
    t_ka  = median(b_ka).time / 1e6   # ms
    t_thr = median(b_thr).time / 1e6
    @printf("  inv_laplace:      KA=%7.2f ms  Threads=%7.2f ms  ratio=%.2f\n",
            t_ka, t_thr, t_ka/t_thr)

    # ── grad_multiply ─────────────────────────────────────────────────────────
    b_ka  = @benchmark grad_multiply_ka!($out, $f, $kx) setup=(out .= 0) evals=1
    b_thr = @benchmark grad_multiply_threads!($out, $f, $kx) setup=(out .= 0) evals=1
    t_ka  = median(b_ka).time / 1e6
    t_thr = median(b_thr).time / 1e6
    @printf("  grad_multiply:    KA=%7.2f ms  Threads=%7.2f ms  ratio=%.2f\n",
            t_ka, t_thr, t_ka/t_thr)

    # ── fmu2_elementwise ──────────────────────────────────────────────────────
    b_ka  = @benchmark fmu2_elementwise_ka!($out, $f, $f2) setup=(out .= 0) evals=1
    b_thr = @benchmark fmu2_elementwise_threads!($out, $f, $f2) setup=(out .= 0) evals=1
    t_ka  = median(b_ka).time / 1e6
    t_thr = median(b_thr).time / 1e6
    @printf("  fmu2_elementwise: KA=%7.2f ms  Threads=%7.2f ms  ratio=%.2f\n",
            t_ka, t_thr, t_ka/t_thr)

    # ── field_add (real) ──────────────────────────────────────────────────────
    f_real  = rand(Float64, res, res, res÷2+1)
    out_real = similar(f_real)
    b_ka  = @benchmark field_add_ka!($out_real, $f_real, 0.5) setup=(out_real .= 0) evals=1
    b_thr = @benchmark field_add_threads!($out_real, $f_real, 0.5) setup=(out_real .= 0) evals=1
    t_ka  = median(b_ka).time / 1e6
    t_thr = median(b_thr).time / 1e6
    @printf("  field_add:        KA=%7.2f ms  Threads=%7.2f ms  ratio=%.2f\n",
            t_ka, t_thr, t_ka/t_thr)

    println()
end

println("=" ^ 70)
println("Full LPT pipeline benchmark (1LPT, backend=:ka vs :threads)")
println("=" ^ 70)
println()

for res in [32, 64]
    c   = Cosmology("Planck18EEBAOSN")
    pk  = linear_power_spectrum(c)
    T   = Float32
    fphi = generate_grf(:fourier, 3, pk, res, 100.0, 42; dtype=T, dtype_c=Complex{T})
    grid = get_fourier_grid(res, 100.0; T=T)

    b_ka  = @benchmark compute_lpt($fphi, $grid; n_order=1, backend=:ka) evals=1
    b_thr = @benchmark compute_lpt($fphi, $grid; n_order=1, backend=:threads) evals=1
    t_ka  = median(b_ka).time / 1e6
    t_thr = median(b_thr).time / 1e6
    @printf("  1LPT res=%3d:  KA=%7.2f ms  Threads=%7.2f ms  ratio=%.2f\n",
            res, t_ka, t_thr, t_ka/t_thr)
end
println()
println("ratio > 1 means KA is slower than Threads (confirms GraphGP.jl finding)")
println("ratio < 1 means KA wins (overhead amortised at large problem size)")
