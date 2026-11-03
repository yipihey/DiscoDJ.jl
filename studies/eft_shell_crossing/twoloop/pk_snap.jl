# Nonlinear P(k) of PM snapshots on the same deposit as sigmaM.jl (2³ resampled sheet, CIC, 384³, window deconvolved) → $EFTSC_SCRATCH/pk_snap/
#   CUDA_VISIBLE_DEVICES="" julia --project=.. -t 8 pk_snap.jl <cfg> <seeds> <signs> <D,D,...>
src = read(joinpath(@__DIR__, "sigmaM.jl"), String); include_string(Main, src[1:findfirst("cfg, seeds, signs, Ds = ARGS", src)[1]-1])
cfg, seeds, signs, Ds = ARGS[1], parse.(Int, split(ARGS[2], ",")), parse.(Int, split(ARGS[3], ",")), parse.(Float64, split(ARGS[4], ","))
L = TOML.parsefile(joinpath(scratch(), cfg, "config.toml"))["L"]; od = joinpath(scratch(), "pk_snap"); mkpath(od)
const EDGES = exp.(range(log(0.012), log(2.4), length=61))
function pk_bins(F, L)                     # a function, not global scope: 28 million modes per snapshot
    kf = 2π / L; nb = length(EDGES) - 1; T = Threads.nthreads()
    wn = [i == 0 ? 1.0 : (x = π * i / NM; (sin(x) / x)^2) for i in 0:NM]; sg(i) = i < NM ÷ 2 ? i : i - NM
    S = [zeros(nb) for _ in 1:T]; C = [zeros(nb) for _ in 1:T]; K = [zeros(nb) for _ in 1:T]; Nl = size(F, 3)
    Threads.@threads :static for t in 1:T
        s_, c_, k_ = S[t], C[t], K[t]
        @inbounds for l in (1 + (t - 1) * Nl ÷ T):(t * Nl ÷ T), j in axes(F, 2), i in axes(F, 1)
            a, b, c = i - 1, sg(j - 1), sg(l - 1); k = kf * sqrt(Float64(a^2 + b^2 + c^2))
            (k < EDGES[1] || k >= EDGES[end]) && continue
            q = searchsortedlast(EDGES, k); W = (i == 1 || i == size(F, 1)) ? 1.0 : 2.0
            s_[q] += W * abs2(F[i, j, l]) / (wn[a + 1] * wn[abs(b) + 1] * wn[abs(c) + 1])^2; c_[q] += W; k_[q] += W * k
        end
    end
    return sum(S), sum(C), sum(K)
end

for s in seeds, sg in signs, D in Ds
    f = joinpath(od, @sprintf("%s_L%d_s%+d_D%.3f.h5", cfg, s, sg, D)); isfile(f) && continue
    Ψ = h5open(h -> read(h["Psi"]), joinpath(scratch(), cfg, "snap", @sprintf("pm_IR_L%d_S0_s%+d_D%.3f.h5", s, sg, D)))
    ρ = cic(resampled(Ψ, L), L); F = rfft(ρ .- 1f0) ./ NM^3; Ψ = nothing
    sum_, cnt, ks = pk_bins(F, L)
    h5open(f * ".tmp", "w") do h
        h["k"] = ks ./ max.(cnt, 1); h["P"] = sum_ ./ max.(cnt, 1) .* L^3; h["n"] = cnt; attrs(h)["D"] = D
    end
    mv(f * ".tmp", f); @printf "%s\n" basename(f); flush(stdout)
end
println("PK DONE")
