using DiscoDJNative, CUDA
using DiscoDJNative: HalfField
gpu_used() = parse(Int, readchomp(`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0`))
res=parse(Int,ARGS[1]); order=parse(Int,ARGS[2]); store=Symbol(length(ARGS)>=3 ? ARGS[3] : "f32")
ff!(::Nothing)=nothing; ff!(a::CuArray)=CUDA.unsafe_free!(a); ff!(h::HalfField)=CUDA.unsafe_free!(h.dev)
freeres!(r)=(ff!(r.psi1);ff!(r.psi2);ff!(r.psi3))
try
    CUDA.unsafe_free!(CUDA.ones(Float32,1))          # init CUDA context
    base = gpu_used()
    grid=get_fourier_grid(res,1000.0;T=Float32); gg=to_gpu(grid)
    fg=CUDA.rand(ComplexF32,res÷2+1,res,res)
    freeres!(CUDA.@sync compute_lpt(fg,gg;n_order=order,backend=:ka,store=store))  # warmup (loads cuFFT)
    GC.gc(true); CUDA.reclaim()                       # clear churn → pool at baseline
    r = CUDA.@sync compute_lpt(fg,gg;n_order=order,backend=:ka,store=store)  # ONE measured compute
    total = maximum(gpu_used() for _ in 1:4)          # retained high-water (stable)
    println("RESULT res=$res order=$order store=$store base_MiB=$base total_MiB=$total")
catch e
    s = sprint(showerror, e)
    oom = (e isa OutOfGPUMemoryError) || occursin("out of memory", lowercase(s)) || occursin("CUFFT", s)
    println("RESULT res=$res order=$order store=$store total_MiB=", oom ? "OOM" : "ERR")
end
