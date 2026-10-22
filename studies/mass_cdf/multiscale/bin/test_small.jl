include(joinpath(@__DIR__, "_env.jl"))
using .MS
include_string(Main, "")
N = 32; L = 500.0; R_f = 18.0; h = L/N
t0=time()
dj0 = M.make_dj(; N, L, R_f)
w = M.white_noise(1, N; N_fine = 256)
dj = M.with_white_noise(dj0, w).with_lpt(; n_order = 4)
println("lpt ", time()-t0)
G1 = M.deformation_gradient(dj, M.lpt_psi(dj, "psi_1"))
A6 = MS.tensor6(G1, N)
println("asym of G1: ", maximum(maximum(abs.(g - g')) for g in G1))
println("nyq frac: ", [MS.nyquist_fraction(a, N) for a in A6])
AC, AP = MS.cube_and_parent(A6, N, L)
println("AP vs avg27(AC): ", maximum(maximum(abs, AP[c] .- MS.avg27(AC[c])) for c in 1:6), "  scale ", maximum(abs, AP[1]))
# own gradient vs deformation_gradient
psi1 = M.pyconvert(Matrix{Float32}, M.np.asarray(M.lpt_psi(dj, "psi_1")).reshape(-1, 3))
g = MS.gradient_from_psi(psi1, N, L)
A6b = MS.sym6_from_grad(g)
println("own grad vs DISCO: ", maximum(maximum(abs, A6[c] .- A6b[c]) for c in 1:6), " asym ", MS.asym_fraction(g))
# delta
d = M.delta_ini(dj); println("rms delta ", sqrt(mean(abs2,d)), " trace check ", maximum(abs, MS.to_xyz(d,N) .+ (A6[1].+A6[2].+A6[3])))
# J cube
ψ = [sum(Float32(ε^n) * M.pyconvert(Matrix{Float32}, M.np.asarray(M.lpt_psi(dj, "psi_$n")).reshape(-1, 3)) for n in 1:4) for ε in (1.0,)][1]
J = MS.cube_jacobian(ψ, N, L)
println("mean J ", mean(J), " min ", minimum(J), " ", size(J))
# compare J cube with det(I+AC) approx
dd = [det(MS.SM3d(I) + MS.mat([AC[c][i,j,k] for c in 1:6],1.0)) for i in 1:N,j in 1:N,k in 1:N]
J1 = MS.cube_jacobian(psi1, N, L)
println("J1 cube vs det(I+AC): corr ", cor(vec(J1), vec(dd)), " rms diff ", sqrt(mean(abs2, J1 .- dd)))
kt, Pt = M.pk_table(dj)
for meth in (:box, :continuum)
    S = MS.predict_sectors(kt, Pt, N, L; method = meth)
    println(meth, " ", MS.ratios(S))
end
mS = MS.sector_moment(AC, AP); println(mS)
