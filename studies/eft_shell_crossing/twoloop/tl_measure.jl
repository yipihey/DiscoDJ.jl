# Experiment 3 measurement driver (CPU).
#
#   julia --project=.. -t 32 tl_measure.jl stress <config> <model> <seed> <sign>     model ∈ pm, lpt1..lpt7
#   julia --project=.. -t 32 tl_measure.jl ops    <config> <seed>
#
# stress: for every snapshot epoch and Λ, the coarse-grained τ̂ (kinetic, gravitational), δ_Λ, π_Λ, curl v_Λ on the low-k
#         modes → $EFTSC_SCRATCH/<config>/tl/stress_<model>_L<seed>_s<sign>.h5
# ops:    the 27 stress functions of arXiv:2509.05187 (A.20) and δ̃^(1..3), θ̃^(1..3) of the W_Λ-filtered linear field of
#         `seed` (sign +1; odd orders flip for −L) → $EFTSC_SCRATCH/<config>/tl/ops_L<seed>.h5
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
include(joinpath(@__DIR__, "StressTL.jl"))
include(joinpath(@__DIR__, "OpsTL.jl"))
using .EFTShell, .StressTL, .OpsTL, DiscoDJNative, FFTW, HDF5, TOML, Printf, Dates
const DJ = DiscoDJNative
FFTW.set_num_threads(parse(Int, get(ENV, "TL_FFTW_THREADS", "1")))   # FFTW's threaded spawnloop segfaults here: keep 1

scratch() = get(ENV, "EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
const LAMBDAS = parse.(Float64, split(get(ENV, "TL_LAMBDAS", "0.1,0.15,0.2,0.3,0.4"), ","))
const KSTORE = parse(Float64, get(ENV, "TL_KSTORE", "0.3"))
const NMESH = parse(Int, get(ENV, "TL_NMESH", "384"))
const NRES = parse(Int, get(ENV, "TL_NRES", "2"))
const FILTER = Symbol(get(ENV, "TL_FILTER", "sharp"))                 # sharp (primary, matched to IR members) | gauss

function state(cfgname, model, seed, sign, D)
    sd = joinpath(scratch(), cfgname, "snap")
    if model == "pm"
        h = h5open(joinpath(sd, @sprintf("pm_IR_L%d_S0_s%+d_D%.3f.h5", seed, sign, D)))
        Ψ = read(h["Psi"]); Π = read(h["Pi"]); close(h)
        return Ψ, Π
    end
    n = parse(Int, model[4:end])
    h = h5open(joinpath(sd, @sprintf("lpt_IR_L%d_S0_s%+d_shapes.h5", seed, sign)))
    ψ = read(h["psi_1"]); Ψ = Float64(D) .* ψ; Π = copy(Float64.(ψ))
    for m in 2:n
        ψ = read(h["psi_$m"]); Ψ .+= D^m .* ψ; Π .+= m * D^(m - 1) .* ψ
    end
    close(h)
    return Float32.(Ψ), Float32.(Π)
end

function do_stress(cfgname, model, seed, sign)
    cfg = TOML.parsefile(joinpath(scratch(), cfgname, "config.toml")); L = cfg["L"]
    out = joinpath(scratch(), cfgname, "tl", @sprintf("stress_%s_%s_L%d_s%+d.h5", FILTER, model, seed, sign)); mkpath(dirname(out))
    md = TLModes(L, KSTORE)
    h5open(out, "w") do h
        h["nvec"] = md.nvec; h["lambdas"] = LAMBDAS; h["epochs"] = Float64.(cfg["snap_epochs"])
        attrs(h)["nmesh"] = NMESH; attrs(h)["nres"] = NRES
        for D in (haskey(ENV, "TL_EPOCHS") ? parse.(Float64, split(ENV["TL_EPOCHS"], ",")) : cfg["snap_epochs"])
            t0 = time()
            Ψ, Π = state(cfgname, model, seed, sign, D)
            X, V = sheet_resample(Ψ, Π, L; nr=NRES); Ψ = nothing; Π = nothing
            M = deposit_moments(X, V, NMESH, L); X = nothing; V = nothing; GC.gc()
            r = StressTL.stress_modes(M, D, L, LAMBDAS, md; filter=FILTER); M = nothing; GC.gc()
            for Λ in LAMBDAS
                g = @sprintf("D%.3f/L%.3f", D, Λ)
                for (k, v) in pairs(r[Λ]); k in (:Nc, :nmin) ? (attrs(h[g])[String(k)] = v) : (h["$g/$k"] = v); end
            end
            @printf "%s %s L%d s%+d D=%.2f  %.1f s\n" Dates.format(now(), "HH:MM:SS") model seed sign D time() - t0; flush(stdout)
        end
    end
end

function do_ops(cfgname, seed)
    cfg = TOML.parsefile(joinpath(scratch(), cfgname, "config.toml")); L = cfg["L"]; N = cfg["N_ic"]
    pk = modify_pk_like(cfg)
    spec = PairSpec(N=N, N_ic=N, L=L, k_split=cfg["k_split"], k_uv=cfg["k_uv"], long_sign=1, seed_L=seed,
                    fixed_amp=get(cfg, "fixed_amp", false))
    φh = make_fphi(spec, pk, :IR; T=Float64)                     # DJ layout ([3,1,2], half axis z)
    kk = kmag_half(N, L)
    δL = irfft(-(kk .^ 2) .* φh, N, [3, 1, 2])                   # real-space linear δ at D = 1
    md = TLModes(L, KSTORE)
    out = joinpath(scratch(), cfgname, "tl", @sprintf("ops_%s_L%d.h5", FILTER, seed)); mkpath(dirname(out))
    h5open(out, "w") do h
        h["nvec"] = md.nvec; h["lambdas"] = LAMBDAS
        h["dL"] = StressTL.extract(rfft(δL), md, N)
        h["pk_k"] = Float64.(pk["k"]); h["pk_P"] = Float64.(pk["Pk"]); attrs(h)["k_uv"] = cfg["k_uv"]
        for Λ in LAMBDAS
            t0 = time()
            ops, δn, θn, resp = build_operators(δL, L, Λ, md; filter=FILTER)
            g = @sprintf("L%.3f", Λ)
            for (k, v) in ops; h["$g/$k"] = v; end
            for n in 1:3; h["$g/delta$n"] = δn[n]; h["$g/theta$n"] = θn[n]; end
            for (lab, r) in resp, (k, v) in pairs(r); h["$g/resp/$lab/$k"] = v; end
            @printf "ops L%d Λ=%.2f  %.1f s\n" seed Λ time() - t0; flush(stdout)
        end
    end
end

# the same P(k) the runs used (EdS shape or ΛCDM; [pk_mod] UV tilt)
function modify_pk_like(cfg)
    s8 = get(get(cfg, "cosmology", Dict()), "sigma8", 0.8105)
    pk = get(get(cfg, "cosmology", Dict()), "lcdm", false) ? lcdm_pk(; sigma8=s8) : planck_pk()
    if haskey(cfg, "pk_mod")
        ks = cfg["pk_mod"]["k_star"]; n = cfg["pk_mod"]["n"]; P0 = DJ._jinterp(log(ks), log.(pk["k"]), pk["Pk"])
        for i in eachindex(pk["k"]); pk["k"][i] > ks && (pk["Pk"][i] = P0 * (pk["k"][i] / ks)^n); end
    end
    return pk
end

if ARGS[1] == "stress"
    do_stress(ARGS[2], ARGS[3], parse(Int, ARGS[4]), parse(Int, ARGS[5]))
elseif ARGS[1] == "ops"
    do_ops(ARGS[2], parse(Int, ARGS[3]))
end
