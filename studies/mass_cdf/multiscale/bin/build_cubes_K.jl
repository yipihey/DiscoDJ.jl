# Variant "A_K": the deformation tensor the Kuhn cube actually sees (edge differences of the vertex displacement, corner-sampled transversally),
#   A_K,ab(k) = i k_a ψ̂_b(k) · sinc(k_a h/2) Π_{c≠a} cos(k_c h/2) · e^{i k·h/2}      (symmetrised),
# versus the cube-average A_C = (i k_a ψ̂_b) Π_c sinc(k_c h/2) e^{i k·h/2}.   julia multiscale/bin/build_cubes_K.jl <R_f> <seed_lo> <seed_hi>
include(joinpath(@__DIR__, "_env.jl"))
using .MS
R_f = parse(Float64, ARGS[1]); lo = parse(Int, ARGS[2]); hi = parse(Int, ARGS[3]); N = 128; L = 1000.0; h = L / N
outdir = joinpath(M.STUDY_ROOT, "data", "multiscale")
dj0 = M.make_dj(; N, L, R_f)
function AK_from_psi(psi::Matrix{Float32}, N, L)
    h = L / N; nyq = N ÷ 2
    ψk = [rfft(Float64.(MS.to_xyz(psi[:, b], N))) for b in 1:3]
    kh = MS.kaxis_r(N, L); kf = MS.kaxis(N, L)
    kvec(a, i1, i2, i3) = a == 1 ? kh[i1] : a == 2 ? kf[i2] : kf[i3]
    A = Dict{Tuple{Int,Int},Array{Float64,3}}()
    for a in 1:3, b in 1:3
        Fk = similar(ψk[b])
        @inbounds for i3 in 1:N, i2 in 1:N, i1 in 1:N÷2+1
            if (i1 - 1 == nyq) || (i2 - 1 == nyq) || (i3 - 1 == nyq)
                Fk[i1, i2, i3] = 0; continue
            end
            k = (kh[i1], kf[i2], kf[i3])
            f = im * k[a] * MS.sincx(k[a] * h / 2)
            for c in 1:3
                c == a && continue
                f *= cos(k[c] * h / 2)
            end
            f *= cis(sum(k) * h / 2)
            Fk[i1, i2, i3] = f * ψk[b][i1, i2, i3]
        end
        A[(a, b)] = irfft(Fk, N)
    end
    return MS.sym6_from_grad(A), A
end
for seed in lo:hi
    fn = joinpath(outdir, @sprintf("cubesK_L%d_N%d_R%d_seed%d.jld2", L, N, R_f, seed))
    isfile(fn) && continue
    t0 = time()
    w = M.white_noise(seed, N; N_fine = 256)
    dj = M.with_white_noise(dj0, w).with_lpt(; n_order = 1)
    psi1 = M.pyconvert(Matrix{Float32}, M.np.asarray(M.lpt_psi(dj, "psi_1")).reshape(-1, 3))
    AK6, Afull = AK_from_psi(psi1, N, L)
    asym = sum(sum(abs2, Afull[(a, b)] .- Afull[(b, a)]) for a in 1:3, b in 1:3) / sum(sum(abs2, Afull[(a, b)]) for a in 1:3, b in 1:3)
    f32(X) = Float32.(reduce(hcat, [vec(x) for x in X])')
    # consistency with the stored cube Jacobian at tiny amplitude is checked in the regression (trace response)
    jldsave(fn; AK = f32(AK6), asym = asym, seed = seed, R_f = R_f)
    @printf("K seed %d R_f %g: %.0fs asym %.2e\n", seed, R_f, time() - t0, asym); flush(stdout)
end
