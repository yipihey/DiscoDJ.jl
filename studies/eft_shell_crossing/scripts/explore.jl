# Exploratory single-pair run (development only)
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, CUDA, DiscoDJNative, Printf, Statistics
const DJ = DiscoDJNative
N = parse(Int, get(ENV,"N","256")); L = parse(Float64, get(ENV,"L","500")); ks = parse(Float64, get(ENV,"KS","0.2")); kuv = parse(Float64, get(ENV,"KUV","1.2"))
nmax = parse(Int, get(ENV,"NMAX","7")); nst = parse(Int, get(ENV,"NSTEPS","100"))
pk = planck_pk(); c = eds_cosmology()
spec = PairSpec(N=N, L=L, k_split=ks, k_uv=kuv, seed_L=1, seed_S=1001)
@printf "band sigma_delta, sigma_psi (D=1): %s ; long: %s\n" band_sigmas(pk, ks, kuv) band_sigmas(pk, 2pi/L, ks)
epochs = [0.02,0.04,0.06,0.08,0.1,0.125,0.15,0.175,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.85,1.0]
ms = ModeSet(N, L, 0.25; to_device=CuArray)
K = to_gpu(nlpt_kernels(N, L; T=Float32)); cache = Dict()
using Serialization
cachef = joinpath(get(ENV,"EFTSC_SCRATCH","/zpool/nvme/data/tabel_scratch/eft_shell_crossing"), "explore_N$(N)_L$(Int(L))_ks$(ks)_kuv$(kuv).jls")
res = isfile(cachef) ? deserialize(cachef) : Dict()
for w in (:UV, :IR)
  haskey(res, (w,:pm)) && continue
  f = CuArray(make_fphi(spec, pk, w; cache=cache))
  t = @elapsed res[(w,:lpt)] = run_lpt(f, K, ms, L, epochs; n_max=nmax)
  println("$w LPT $(round(t,digits=1)) s"); GC.gc(); CUDA.reclaim()
  r = run_pm(c, f, K, ms, L, epochs; D_ini=0.02, n_steps=nst, mesh_factor=2)
  res[(w,:pm)] = r; println("$w PM $(round(r.seconds,digits=1)) s"); GC.gc(); CUDA.reclaim()
end
serialize(cachef, res)
edges = collect(0.5:1:16) .* (2pi/L)
kv = ms.kvec; km = ms.kmag
xL = longitudinal(res[(:UV,:lpt)].psi[1,:,:], kv)  # psi_1 (shared)
sel = km .< ks; @show maximum(abs.(xL[sel] .- longitudinal(res[(:IR,:lpt)].psi[1,:,:], kv)[sel]))
println("inverted fraction UV: LPT n=1..$nmax and PM")
for (e,D) in enumerate(epochs)
  @printf "D=%.3f " D; for n in 1:nmax; @printf "%.4f " res[(:UV,:lpt)].inv[e,n]; end; @printf "| PM %.4f  IR-PM %.4f\n" res[(:UV,:pm)].inv[e] res[(:IR,:pm)].inv[e]
end
fitC2(A) = (r = response(A); σ = abs.(r.R) .* 0 .+ 1 ./ sqrt.(r.n); fit_counterterm(r.k2, r.k4, r.R, σ; kmax=0.08))
println("C2 (Mpc/h)^2 from displacement, kmax=0.08: LPT n=1..$nmax | PM")
for (e,D) in enumerate(epochs)
  @printf "D=%.3f " D
  for n in 1:nmax
    Δ = sum(D^m .* (res[(:UV,:lpt)].psi[m,:,:] .- res[(:IR,:lpt)].psi[m,:,:]) for m in 1:n)
    @printf "%8.3f " fitC2(accum!(longitudinal(Δ,kv), D .* xL, km, edges)).C2
  end
  if D > 0.02
    Δ = res[(:UV,:pm)].psi[e,:,:] .- res[(:IR,:pm)].psi[e,:,:]
    @printf "| %8.3f" fitC2(accum!(longitudinal(Δ,kv), D .* xL, km, edges)).C2
    @printf "  dens: %8.3f" fitC2(accum!(res[(:UV,:pm)].dens[e,:] .- res[(:IR,:pm)].dens[e,:], D .* (-im .* km .* xL), km, edges)).C2
    @printf "  dens LPT1: %8.3f" fitC2(accum!(res[(:UV,:lpt)].dens[e,1,:] .- res[(:IR,:lpt)].dens[e,1,:], D .* (-im .* km .* xL), km, edges)).C2
  end
  println()
end
println("PM momentum ", res[(:UV,:pm)].mom[end,:])
