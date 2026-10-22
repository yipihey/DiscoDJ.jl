# Per-seed cube features for the Gaussian LPT ensembles:  julia -t 4 --project=studies/mass_cdf multiscale/bin/build_cubes.jl <R_f> <seed_lo> <seed_hi> [N=128] [L=1000]
# Writes data/multiscale/cubes_L<L>_N<N>_R<R_f>_seed<s>.jld2 : AC, AP (6,n) Float32 (ε=1 units; comps xx,yy,zz,xy,xz,yz),
# J_<n>LPT_e<ε> (n=2,4) cube Jacobians, flattening = vec of the [i,j,k] array (i fastest), plus sector second moments.
include(joinpath(@__DIR__, "_env.jl"))
using .MS
R_f = parse(Float64, ARGS[1]); lo = parse(Int, ARGS[2]); hi = parse(Int, ARGS[3])
N = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 128
L = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1000.0
outdir = joinpath(M.STUDY_ROOT, "data", "multiscale"); mkpath(outdir)
dj0 = M.make_dj(; N, L, R_f)
q_py = M.grid_positions_py(N, L)
psi_mat(dj, key) = M.pyconvert(Matrix{Float32}, M.np.asarray(M.lpt_psi(dj, key)).reshape(-1, 3))
for seed in lo:hi
    fn = joinpath(outdir, @sprintf("cubes_L%d_N%d_R%d_seed%d.jld2", L, N, R_f, seed))
    isfile(fn) && (println("exists $fn"); continue)
    t0 = time()
    w = M.white_noise(seed, N; N_fine = 256)
    dj = M.with_white_noise(dj0, w).with_lpt(; n_order = 4)
    G1 = M.deformation_gradient(dj, M.lpt_psi(dj, "psi_1"))
    A6 = MS.tensor6(G1, N)
    AC, AP = MS.cube_and_parent(A6, N, L)
    chk = maximum(maximum(abs, AP[c] .- MS.avg27(AC[c])) for c in 1:6)
    sec = Dict("CC" => MS.sector_moment(AC, AC), "PP" => MS.sector_moment(AP, AP), "CP" => MS.sector_moment(AC, AP),
               "vertex" => MS.sector_moment(A6, A6))
    ψ = [psi_mat(dj, "psi_$n") for n in 1:4]
    epss = R_f == 18 ? (seed == 1 ? [1.0, 0.5, 0.2, 0.1] : [1.0, 0.5]) : [1.0]
    Js = Dict{String,Vector{Float32}}(); meanJ = Dict{String,Float64}()
    for ε in epss, n in (2, 4)
        ψt = sum(Float32(ε^m) .* ψ[m] for m in 1:n)
        J = MS.cube_jacobian(ψt, N, L; q_py)
        key = "J_$(n)LPT_e$(ε)"
        Js[key] = vec(J); meanJ[key] = mean(Float64.(J))
    end
    f32(X) = Float32.(reduce(hcat, [vec(x) for x in X])')            # (6, n)
    jldsave(fn; AC = f32(AC), AP = f32(AP), Js = Js, meanJ = meanJ, sec = sec, N = N, L = L, R_f = R_f, seed = seed,
            sigma_vertex = sqrt(mean(abs2, Float64.(M.delta_ini(dj)))), avg27_check = chk, epss = epss)
    @printf("seed %d R_f %g: %.0fs  avg27 check %.1e  meanJ %s\n", seed, R_f, time() - t0, chk, join((@sprintf("%s=%.6f", k, v) for (k, v) in sort(collect(meanJ))), " "))
    flush(stdout)
end
