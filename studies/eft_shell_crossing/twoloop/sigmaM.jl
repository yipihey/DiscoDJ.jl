# Fixed-mass sigmas of PM snapshots or nLPT (light CPU) → $EFTSC_SCRATCH/sigmaM/<cfg>_<model>_L<seed>_s<sign>_D<D>.h5; analysis in sigmaM_analysis.py
#
#   CUDA_VISIBLE_DEVICES="" julia --project=.. -t 8 sigmaM.jl <cfg> <seeds> <signs> <D,D,...> [model]    model = pm (default) or lpt<n>, summed from the stored displacement shapes
#
# Eulerian: μ_R(x) by top-hat convolution of the CIC density of the 2³-resampled sheet on a 384³ mesh, for 64 radii.
#   count[j, b]  histogram of log μ_R over mesh nodes (random centres)
#   massw[j, b]  the same weighted by ρ at the node (centres drawn from the density)
#   so P(R_M ≤ R) = P(μ_R ≥ M/ρ̄V_R) gives the distribution of the volume enclosing M.
# Lagrangian: V_patch(q; R_L) = image volume of the Lagrangian sphere of radius R_L (mass M = ρ̄V(R_L)): signed P1 tetrahedron volumes (the volume enclosed
#   by the image of the surface, boundary-only) and unsigned (total sheet volume, counts folds), each top-hat filtered on the Lagrangian lattice.
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, HDF5, FFTW, Printf, Statistics, TOML
FFTW.set_num_threads(min(Threads.nthreads(), 16))
scratch() = get(ENV, "EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
const NM = 384; const NB = 2080; const LMIN = -6.0; const LMAX = 7.0
const RE = exp.(range(log(4.0), log(120.0), length=64)); const RL = exp.(range(log(6.0), log(80.0), length=24))
tophat(x) = x < 1e-3 ? 1 - x^2 / 10 : 3 * (sin(x) - x * cos(x)) / x^3

function resampled(Ψ, L; nr=2)
    N = size(Ψ, 1); Np = N^3; X = Array{Float32}(undef, 3, nr^3 * Np)
    kh = [2π / L * i for i in 0:N÷2]; kf = [2π / L * (i < N ÷ 2 ? i : i - N) for i in 0:N-1]
    F = [rfft(Float64.(view(Ψ, :, :, :, c))) for c in 1:3]; s = 0
    for a in 0:nr-1, b in 0:nr-1, c in 0:nr-1
        d = (a, b, c) ./ nr .* (L / N)
        ph = [cis(kh[i] * d[1] + kf[j] * d[2] + kf[l] * d[3]) for i in eachindex(kh), j in eachindex(kf), l in eachindex(kf)]
        for e in 1:3
            ψ = irfft(F[e] .* ph, N)
            Threads.@threads for l in 1:N
                @inbounds for j in 1:N, i in 1:N
                    p = s * Np + i + N * (j - 1) + N^2 * (l - 1); q = ((i - 1) * L / N, (j - 1) * L / N, (l - 1) * L / N)[e] + d[e]
                    X[e, p] = mod(q + ψ[i, j, l], L)
                end
            end
        end
        s += 1
    end
    return X
end

function cic(X, L)
    T = Threads.nthreads(); M = [zeros(Float32, NM, NM, NM) for _ in 1:T]; Np = size(X, 2)
    Threads.@threads :static for t in 1:T
        m = M[t]
        @inbounds for p in (1 + (t - 1) * Np ÷ T):(t * Np ÷ T)
            gx = X[1, p] / L * NM; gy = X[2, p] / L * NM; gz = X[3, p] / L * NM
            ix = floor(Int, gx); iy = floor(Int, gy); iz = floor(Int, gz); fx = gx - ix; fy = gy - iy; fz = gz - iz
            for oz in 0:1, oy in 0:1, ox in 0:1
                w = (ox == 0 ? 1 - fx : fx) * (oy == 0 ? 1 - fy : fy) * (oz == 0 ? 1 - fz : fz)
                m[mod(ix + ox, NM) + 1, mod(iy + oy, NM) + 1, mod(iz + oz, NM) + 1] += Float32(w)
            end
        end
    end
    ρ = sum(M); ρ .*= Float32(NM^3 / Np); return ρ
end

function filt(F, N, L, R)
    kx = [2π / L * i for i in 0:N÷2]; ky = [2π / L * (i < N ÷ 2 ? i : i - N) for i in 0:N-1]; G = similar(F)
    Threads.@threads for l in axes(F, 3)
        @inbounds for j in axes(F, 2), i in axes(F, 1)
            G[i, j, l] = F[i, j, l] * Float32(tophat(R * sqrt(kx[i]^2 + ky[j]^2 + ky[l]^2)))
        end
    end
    return irfft(G, N)
end

function eulerian(ρ, L)
    F = rfft(ρ); cnt = zeros(length(RE), NB); mw = zeros(length(RE), NB); mom = zeros(length(RE), 4)
    db = (LMAX - LMIN) / NB; T = Threads.nthreads()
    for (j, R) in enumerate(RE)
        μ = filt(F, NM, L, R)
        hc = [zeros(NB) for _ in 1:T]; hm = [zeros(NB) for _ in 1:T]; mm = [zeros(4) for _ in 1:T]; N3 = length(μ)
        Threads.@threads :static for t in 1:T
            a = hc[t]; b = hm[t]; m = mm[t]
            @inbounds for p in (1 + (t - 1) * N3 ÷ T):(t * N3 ÷ T)
                v = Float64(max(μ[p], 1f-6)); w = Float64(ρ[p]); d = Float64(μ[p]) - 1
                i = clamp(floor(Int, (log(v) - LMIN) / db) + 1, 1, NB); a[i] += 1; b[i] += w
                m[1] += d; m[2] += d * d; m[3] += w * d; m[4] += w * d * d
            end
        end
        cnt[j, :] .= sum(hc); mw[j, :] .= sum(hm); mom[j, :] .= sum(mm) ./ N3
    end
    return cnt, mw, mom
end

function lagrangian(Ψ, L)
    N = size(Ψ, 1); dq = L / N; S = zeros(Float32, N, N, N); U = zeros(Float32, N, N, N)
    corners = ((0,0,0),(1,0,0),(1,1,0),(1,1,1)), ((0,0,0),(1,0,0),(1,0,1),(1,1,1)), ((0,0,0),(0,1,0),(1,1,0),(1,1,1)), ((0,0,0),(0,1,0),(0,1,1),(1,1,1)), ((0,0,0),(0,0,1),(1,0,1),(1,1,1)), ((0,0,0),(0,0,1),(0,1,1),(1,1,1))
    pos(i, j, l, o) = (x = (dq * (o[1]) + Ψ[mod(i - 1 + o[1], N) + 1, mod(j - 1 + o[2], N) + 1, mod(l - 1 + o[3], N) + 1, 1],
                            dq * (o[2]) + Ψ[mod(i - 1 + o[1], N) + 1, mod(j - 1 + o[2], N) + 1, mod(l - 1 + o[3], N) + 1, 2],
                            dq * (o[3]) + Ψ[mod(i - 1 + o[1], N) + 1, mod(j - 1 + o[2], N) + 1, mod(l - 1 + o[3], N) + 1, 3]); x)
    sgn = (+1, -1, -1, +1, +1, -1)     # orientation of the six tetrahedra so the undeformed cube gives +dq³
    Threads.@threads for l in 1:N
        @inbounds for j in 1:N, i in 1:N
            s = 0.0; u = 0.0
            for (t, c) in enumerate(corners)
                a = pos(i, j, l, c[1]); b = pos(i, j, l, c[2]); d = pos(i, j, l, c[3]); e = pos(i, j, l, c[4])
                u1 = (b[1] - a[1], b[2] - a[2], b[3] - a[3]); u2 = (d[1] - a[1], d[2] - a[2], d[3] - a[3]); u3 = (e[1] - a[1], e[2] - a[2], e[3] - a[3])
                v = sgn[t] * (u1[1] * (u2[2] * u3[3] - u2[3] * u3[2]) - u1[2] * (u2[1] * u3[3] - u2[3] * u3[1]) + u1[3] * (u2[1] * u3[2] - u2[2] * u3[1])) / 6
                s += v; u += abs(v)
            end
            S[i, j, l] = Float32(s / dq^3); U[i, j, l] = Float32(u / dq^3)
        end
    end
    FS = rfft(S); FU = rfft(U); out = zeros(length(RL), 6)
    for (j, R) in enumerate(RL)
        vs = filt(FS, N, L, R); vu = filt(FU, N, L, R)
        ok = count(>(0f0), vs) / length(vs); ls = log.(max.(vs, 1f-3)); lu = log.(max.(vu, 1f-3))     # Fourier top-hat ringing can take the filtered unsigned volume slightly below zero
        out[j, :] .= (mean(ls), std(ls), mean(lu), std(lu), ok, mean(Float64.(vs)))
    end
    return out, mean(Float64.(S)), mean(Float64.(U))
end

cfg, seeds, signs, Ds = ARGS[1], parse.(Int, split(ARGS[2], ",")), parse.(Int, split(ARGS[3], ",")), parse.(Float64, split(ARGS[4], ","))
L = TOML.parsefile(joinpath(scratch(), cfg, "config.toml"))["L"]; od = joinpath(scratch(), "sigmaM"); mkpath(od)
model = length(ARGS) >= 5 ? ARGS[5] : "pm"
function displacement(cfg, model, s, sg, D)
    sd = joinpath(scratch(), cfg, "snap")
    model == "pm" && return h5open(h -> read(h["Psi"]), joinpath(sd, @sprintf("pm_IR_L%d_S0_s%+d_D%.3f.h5", s, sg, D)))
    startswith(model, "lpt") || error("unknown model $model")
    n = parse(Int, model[4:end])
    h5open(joinpath(sd, @sprintf("lpt_IR_L%d_S0_s%+d_shapes.h5", s, sg))) do h
        Ψ = Float64(D) .* read(h["psi_1"])
        for m in 2:n
            Ψ .+= D^m .* read(h["psi_$m"])
        end
        Float32.(Ψ)
    end
end
for s in seeds, sg in signs, D in Ds
    tag = @sprintf("%s_%s_L%d_s%+d_D%.3f", cfg, model, s, sg, D); f = joinpath(od, tag * ".h5"); isfile(f) && continue
    t0 = time(); Ψ = displacement(cfg, model, s, sg, D)
    X = resampled(Ψ, L); ρ = cic(X, L); X = nothing; t1 = time()
    cnt, mw, mom = eulerian(ρ, L); t2 = time()
    lag, ms, mu = lagrangian(Ψ, L); t3 = time()
    h5open(f * ".tmp", "w") do h
        h["RE"] = RE; h["RL"] = RL; h["count"] = cnt; h["massw"] = mw; h["mom"] = mom; h["lag"] = lag
        h["lbins"] = collect(range(LMIN, LMAX, length=NB + 1)); attrs(h)["D"] = D; attrs(h)["mean_signed"] = ms; attrs(h)["mean_unsigned"] = mu; attrs(h)["mean_rho"] = mean(Float64.(ρ))
    end
    mv(f * ".tmp", f)
    @printf "%s  deposit %.0f s, eulerian %.0f s, lagrangian %.0f s; <rho>-1 %.1e <V_signed>-1 %.1e\n" tag t1 - t0 t2 - t1 t3 - t2 mean(Float64.(ρ)) - 1 ms - 1; flush(stdout)
    ρ = nothing; GC.gc()
end
println("SIGMAM DONE")
