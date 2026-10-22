# Cube features for the FEM seed-1 runs:  julia -t 4 --project=studies/mass_cdf multiscale/bin/build_fem_cubes.jl <N>   (L=500, R_f=18, EdS + planck18 P(k) table, as in src/fem.jl)
include(joinpath(@__DIR__, "_env.jl"))
using .MS
N = parse(Int, ARGS[1]); L = 500.0; R_f = 18.0; seed = 1
h = L / N
dirname_ = joinpath(M.STUDY_ROOT, "data", "FEM_L500_N$N")
files = Dict(ε => joinpath(dirname_, N == 64 ? "part_fem_1-10_elements_seed1_eps$(ε).jld2" : "part_fem_1-3_elements_seed1_eps$(ε).jld2") for ε in (-1.0, -0.5, 1.0, 0.5))
all(isfile, values(files)) || error("missing FEM element files: $(filter(!isfile, collect(values(files))))")
outdir = joinpath(M.STUDY_ROOT, "data", "multiscale"); mkpath(outdir)
Ωb = 0.0488911
eds = Dict("Omega_c" => 1.0 - Ωb, "Omega_b" => Ωb, "Omega_k" => 0.0, "h" => 0.67742, "n_s" => 0.96822, "sigma8" => 0.8105, "w0" => -1.0, "wa" => 0.0)
pk_file = joinpath(M.STUDY_ROOT, "data/pk/planck18_pk_z0.txt")
dj0 = M.make_dj(; N, L, R_f, cosmo = eds, pk_file)
q_py = M.grid_positions_py(N, L)
psi_mat(dj, key) = M.pyconvert(Matrix{Float32}, M.np.asarray(M.lpt_psi(dj, key)).reshape(-1, 3))
to_mesh(ψ::Matrix{Float32}) = M.jnp.asarray(M.np.asarray(ψ)).reshape(N, N, N, 3)
t0 = time()
w = M.white_noise(seed, N; N_fine = 256)
dj1 = M.with_white_noise(dj0, w).with_lpt(; n_order = 4)
ψn = [psi_mat(dj1, "psi_$n") for n in 1:4]
d1 = JLD2.load(files[1.0])
ψ_lin = d1["psi_lin_fem"]
G_lin = M.deformation_gradient(dj1, to_mesh(ψ_lin))
δ_fem = Float32[-tr(g) for g in G_lin]
G_in = M.deformation_gradient(dj1, M.lpt_psi(dj1, "psi_1"))
f32(X) = Float32.(reduce(hcat, [vec(x) for x in X])')
A6_in = MS.tensor6(G_in, N); A6_lin = MS.tensor6(G_lin, N)
ACi, APi = MS.cube_and_parent(A6_in, N, L); ACl, APl = MS.cube_and_parent(A6_lin, N, L)
asym = mean(g -> sum(abs2, g - g'), G_lin) / mean(g -> sum(abs2, g), G_lin)
println("setup ", time() - t0, " s;  rms delta_in ", sqrt(mean(abs2, M.delta_ini(dj1))), " rms delta_fem_lin ", sqrt(mean(abs2, δ_fem)), " asym ", asym)
djm = dj0.with_external_ics(; delta = M.np.asarray(δ_fem).reshape(N, N, N)).with_lpt(; n_order = 4)
ψm = [psi_mat(djm, "psi_$n") for n in 1:4]
Js = Dict{String,Vector{Float32}}(); chk = Dict{String,Float64}(); meanJ = Dict{String,Float64}()
for ε in (-1.0, -0.5, 0.5, 1.0)
    d = JLD2.load(files[ε])
    Jf = MS.cube_jacobian(d["psi_fem"], N, L; q_py)
    Jt = MS.cube_J_from_tets(d["Jtet_FEM"], N)
    chk["FEM_e$ε"] = maximum(abs, Jf .- Jt)
    Js["FEM_e$ε"] = vec(Jf); meanJ["FEM_e$ε"] = mean(Float64.(Jf))
    for (tag, ψs) in (("raw", ψn), ("matched", ψm)), n in (2, 4)
        J = MS.cube_jacobian(sum(Float32(ε^m) .* ψs[m] for m in 1:n), N, L; q_py)
        Js["J_$(n)LPT_$(tag)_e$ε"] = vec(J); meanJ["J_$(n)LPT_$(tag)_e$ε"] = mean(Float64.(J))
    end
    @printf("N=%d eps=%+.1f: mean J_FEM %.6f, max|Jcube(pos) - Jcube(tet)| = %.2e, minJ_FEM %.3f\n", N, ε, meanJ["FEM_e$ε"], chk["FEM_e$ε"], minimum(Jf))
end
sec = Dict("lin" => Dict("CC" => MS.sector_moment(ACl, ACl), "PP" => MS.sector_moment(APl, APl), "CP" => MS.sector_moment(ACl, APl)),
           "in" => Dict("CC" => MS.sector_moment(ACi, ACi), "PP" => MS.sector_moment(APi, APi), "CP" => MS.sector_moment(ACi, APi)))
jldsave(joinpath(outdir, "cubes_FEM_L500_N$N.jld2"); AC_lin = f32(ACl), AP_lin = f32(APl), AC_in = f32(ACi), AP_in = f32(APi),
        Js = Js, meanJ = meanJ, tetcheck = chk, sec = sec, N = N, L = L, R_f = R_f, asym_Glin = asym)
println("done N=$N in ", time() - t0, " s")
