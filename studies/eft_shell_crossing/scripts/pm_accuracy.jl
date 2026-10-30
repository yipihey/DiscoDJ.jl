# PM convergence before shell crossing (development diagnostic): one fixed realization (white noise
# on the 256³ base grid), PM at several particle grids N and mesh factors, compared mode by mode with
# 7LPT computed on the base grid — band transfer T(k) of the UV member, and the response C₂.
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, CUDA, DiscoDJNative, Printf, Statistics, Serialization
const DJ = DiscoDJNative
L = parse(Float64, get(ENV,"L","500")); ks = parse(Float64, get(ENV,"KS","0.2")); kuv = parse(Float64, get(ENV,"KUV","1.2"))
pk = planck_pk(); c = eds_cosmology()
base = PairSpec(N=256, N_ic=256, L=L, k_split=ks, k_uv=kuv, seed_L=1, seed_S=1001)
epochs = [0.1, 0.15, 0.2]
kst = 1.25
cache = Dict()
function lpt_ref()
  ms = ModeSet(256, L, kst; to_device=CuArray); K = to_gpu(nlpt_kernels(256, L; T=Float32))
  r = Dict(w => run_lpt(CuArray(make_fphi(base, pk, w; cache=cache)), K, ms, L, epochs; n_max=7) for w in (:UV,:IR))
  return r, ms
end
ref, ms0 = lpt_ref(); GC.gc(); CUDA.reclaim()
edges_b = [0.05, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25]
edges_r = collect(0.5:1:8) .* (2pi/L)
lowk = ms0.kmag .<= 8.5*2pi/L
xL = longitudinal(ref[:UV].psi[1,:,:], ms0.kvec)
lptD(w, D, n) = sum(D^m .* ref[w].psi[m,:,:] for m in 1:n)
function c2(Δ, D)
  sel = lowk
  A = accum!(longitudinal(Δ[sel,:], ms0.kvec[sel,:]), D .* xL[sel], ms0.kmag[sel], edges_r)
  r = response(A); fit_counterterm(r.k2, r.k4, r.R, 1 ./ sqrt.(r.n); kmax=0.08).C2
end
@printf "LPT7 C2: %s\n" join([@sprintf("%.4f", c2(lptD(:UV,D,7) .- lptD(:IR,D,7), D)) for D in epochs], " ")
@printf "LPT3 C2: %s\n" join([@sprintf("%.4f", c2(lptD(:UV,D,3) .- lptD(:IR,D,3), D)) for D in epochs], " ")
configs = [(256,1),(256,2),(384,1),(384,2),(512,1),(512,2)]
sel = get(ENV,"CONFIGS","")
for (N, mf) in configs
  (sel != "" && !occursin("$(N)x$(mf)", sel)) && continue
  spec = PairSpec(base; N=N)
  ms = ModeSet(N, L, kst; to_device=CuArray); @assert ms.nvec == ms0.nvec
  K = to_gpu(nlpt_kernels(N, L; T=Float32))
  pm = Dict()
  t = @elapsed for w in (:UV, :IR)
    pm[w] = run_pm(c, CuArray(make_fphi(spec, pk, w; cache=cache)), K, ms, L, epochs; D_ini=0.02, n_steps=100, mesh_factor=mf)
    GC.gc(); CUDA.reclaim()
  end
  e = 2; D = epochs[e]
  # band transfer of the UV displacement vs 7LPT
  y = pm[:UV].psi[e,:,:]; x = lptD(:UV, D, 7)
  bi = kbin_index(ms0.kmag, edges_b); T = zeros(length(edges_b)-1); B = similar(T)
  for m in eachindex(bi); b = bi[m]; b == 0 && continue; T[b] += real(sum(y[m,:] .* conj.(x[m,:]))); B[b] += sum(abs2, x[m,:]); end
  @printf "PM N=%d mesh=%dN (%5.0fs): T_band(D=%.2f)= %s | C2(D)= %s\n" N mf t D join([@sprintf("%.4f", v) for v in T./B], " ") join([@sprintf("%.4f", c2(pm[:UV].psi[i,:,:] .- pm[:IR].psi[i,:,:], epochs[i])) for i in 1:3], " ")
  serialize(joinpath("/zpool/nvme/data/tabel_scratch/eft_shell_crossing", "pmacc_N$(N)_m$(mf).jls"), pm)
  pm = nothing; GC.gc(); CUDA.reclaim()
end
