# Part 5: compressed per-cube arrays for seed 1 (R_f=18 Gaussian) and the FEM seed.  Layout: tensors (6,N,N,N) [comp,i,j,k] with comps xx,yy,zz,xy,xz,yz;
# scalars (N,N,N) [i,j,k]; eps=1 units for A, eta; J at the stated eps.  eta_C = A_C - beta A_P sector-wise (trace beta0, diag-traceless betaE, off-diag betaT) from the
# Gaussian box prediction.
include(joinpath(@__DIR__, "_env.jl"))
using .MS
datadir = joinpath(M.STUDY_ROOT, "data/multiscale")
Ωb = 0.0488911
eds = Dict("Omega_c" => 1.0 - Ωb, "Omega_b" => Ωb, "Omega_k" => 0.0, "h" => 0.67742, "n_s" => 0.96822, "sigma8" => 0.8105, "w0" => -1.0, "wa" => 0.0)
pk_file = joinpath(M.STUDY_ROOT, "data/pk/planck18_pk_z0.txt")
function betas(L, N, R_f, kind)
    dj0 = kind == :lcdm ? M.make_dj(; N, L, R_f) : M.make_dj(; N, L, R_f, cosmo = eds, pk_file)
    kt, Pt = M.pk_table(dj0); MS.ratios(MS.predict_sectors(kt, Pt, N, L; method = :box)).β
end
function eta(AC::Matrix{Float32}, AP::Matrix{Float32}, β)
    n = size(AC, 2); out = Matrix{Float32}(undef, 6, n)
    for e in 1:n
        c = Float64.(AC[:, e]); p = Float64.(AP[:, e])
        tC = (c[1] + c[2] + c[3]) / 3; tP = (p[1] + p[2] + p[3]) / 3
        for i in 1:3
            out[i, e] = (tC - β[1] * tP) + ((c[i] - tC) - β[2] * (p[i] - tP))
        end
        for i in 4:6; out[i, e] = c[i] - β[3] * p[i]; end
    end
    return out
end
t6(A, N) = reshape(A, 6, N, N, N)
s3(v, N) = reshape(v, N, N, N)
dl(AC) = -(AC[1, :] .+ AC[2, :] .+ AC[3, :])
# ---- Gaussian seed 1, R_f = 18
N = 128; L = 1000.0
d = JLD2.load(joinpath(datadir, "cubes_L1000_N128_R18_seed1.jld2")); β = betas(L, N, 18.0, :lcdm)
out = Dict{String,Any}("A_C" => t6(d["AC"], N), "A_P" => t6(d["AP"], N), "eta_C" => t6(eta(d["AC"], d["AP"], β), N), "delta_L_cube" => s3(dl(d["AC"]), N),
                       "beta0_E_T" => collect(β), "N" => N, "L" => L, "R_f" => 18.0, "seed" => 1, "layout" => "tensors (6,N,N,N)[comp,i,j,k]; scalars (N,N,N)[i,j,k]; A in eps=1 units")
for (k, v) in d["Js"]; out[k] = s3(v, N); end
jldsave(joinpath(datadir, "arrays_L1000_N128_R18_seed1.jld2"), true; (Symbol(k) => v for (k, v) in out)...)
println("saved Gaussian arrays")
# ---- FEM seed 1
for N in (64, 128)
    fn = joinpath(datadir, "cubes_FEM_L500_N$N.jld2"); isfile(fn) || continue
    d = JLD2.load(fn); β = betas(500.0, N, 18.0, :eds)
    out = Dict{String,Any}("A_C_lin" => t6(d["AC_lin"], N), "A_P_lin" => t6(d["AP_lin"], N), "eta_C_lin" => t6(eta(d["AC_lin"], d["AP_lin"], β), N),
                           "A_C_in" => t6(d["AC_in"], N), "A_P_in" => t6(d["AP_in"], N), "eta_C_in" => t6(eta(d["AC_in"], d["AP_in"], β), N),
                           "delta_L_cube_lin" => s3(dl(d["AC_lin"]), N), "delta_L_cube_in" => s3(dl(d["AC_in"]), N),
                           "beta0_E_T" => collect(β), "N" => N, "L" => 500.0, "R_f" => 18.0, "seed" => 1,
                           "layout" => "tensors (6,N,N,N)[comp,i,j,k]; scalars (N,N,N)[i,j,k]; A in eps=1 units; lin = from FEM psi_lin, in = input linear field; J keys carry eps")
    for (k, v) in d["Js"]; out[k] = s3(v, N); end
    jldsave(joinpath(datadir, "arrays_FEM_L500_N$(N)_seed1.jld2"), true; (Symbol(k) => v for (k, v) in out)...)
    println("saved FEM N=$N arrays")
end
