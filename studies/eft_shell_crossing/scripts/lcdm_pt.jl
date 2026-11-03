# Field-level second-order PT for the ΛCDM counterterm calibration (CPU).
#
#   julia --project=. -t 16 scripts/lcdm_pt.jl config/lcdm_full.toml
#
# For every long seed of the config, rebuild the (fixed-amplitude) linear field δ_L at D = 1 exactly as the runs do,
# and compute the SPT second-order field δ⁽²⁾ = 5/7 δ² + ∇δ·∇∇⁻²δ + 2/7 (∂ᵢ∂ⱼ∇⁻²δ)² on a 3/2-padded grid (no aliasing),
# reduced to the config's stored mode set.  P22 of *this* realization is then L³|δ⁽²⁾_k|² D⁴, so the one-loop comparison
# with the PM auto-power carries the same realization of the mode coupling (δ⁽²⁾ is even in δ_L: identical for ±L).
# Output: results/<name>_pt.h5 with nvec, dL (M, seeds), d2 (M, seeds) as discrete Fourier coefficients (rfft/N³).
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, DiscoDJNative, FFTW, HDF5, TOML, Printf
const DJ = DiscoDJNative
FFTW.set_num_threads(min(16, Threads.nthreads()))

function second_order_modes(fφ::Array{ComplexF64,3}, N::Int, L::Real, ms::ModeSet)
    kf = DJ._kfull(N, Float64(L)); kh = DJ._khalf(N, Float64(L))
    kv = (reshape(kf, :, 1, 1), reshape(kf, 1, :, 1), reshape(kh, 1, 1, :))
    k2 = @. kv[1]^2 + kv[2]^2 + kv[3]^2
    δh = @. -k2 * fφ                                   # ∇²φ = δ
    ir(x) = irfft(x, N, [3, 1, 2])
    δ = ir(δh)
    δ2 = (5 / 7) .* δ .^ 2
    for i in 1:3
        δ2 .+= ir(@. im * kv[i] * δh) .* ir(@. im * kv[i] * fφ)
    end
    for i in 1:3, j in i:3
        t = ir(@. -kv[i] * kv[j] * fφ)
        δ2 .+= (i == j ? 2 / 7 : 4 / 7) .* t .^ 2
    end
    return field_modes(ms, δ), field_modes(ms, δ2)
end

function main(args)
    cfg = TOML.parsefile(args[1])
    s8 = get(get(cfg, "cosmology", Dict()), "sigma8", 0.8105)
    pk = lcdm_pk(; sigma8=s8)
    N = cfg["N_ic"]; L = cfg["L"]; Np = 3N ÷ 2
    ms = ModeSet(Np, L, cfg["k_store"])
    seeds = cfg["seeds"]["long"]
    dL = zeros(ComplexF64, length(ms), length(seeds)); d2 = similar(dL)
    for (is, sL) in enumerate(seeds)
        spec = PairSpec(N=N, N_ic=N, L=L, k_split=cfg["k_split"], k_uv=cfg["k_uv"], long_sign=1, seed_L=sL,
                        fixed_amp=get(cfg, "fixed_amp", false))
        f = embed_fphi(make_fphi(spec, pk, :IR; T=Float64), Np; T=Float64)
        t = @elapsed (a, b) = second_order_modes(f, Np, L, ms)
        dL[:, is] .= a; d2[:, is] .= b
        @printf "seed %d: %.1f s   P22/P at k≈0.1: %.4f\n" sL t (m = abs.(ms.kmag .- 0.1) .< 0.005; sum(abs2, b[m]) / sum(abs2, a[m]))
        flush(stdout)
    end
    out = joinpath(@__DIR__, "..", "results", cfg["name"] * "_pt.h5")
    h5open(out, "w") do h
        h["nvec"] = ms.nvec; h["seeds"] = collect(seeds); h["dL"] = dL; h["d2"] = d2
        h["pk_k"] = Float64.(pk["k"]); h["pk_P"] = Float64.(pk["Pk"])   # linear P(k) at z = 0 used by the ICs
        attrs(h)["config"] = sprint(TOML.print, cfg); attrs(h)["sigma8"] = s8
    end
    println("wrote ", out)
end
main(ARGS)
