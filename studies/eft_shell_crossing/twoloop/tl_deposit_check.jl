# Is the CIC moment deposit of the stress pipeline biased at low k?  Direct particle sums (1/Np) Σ w e^{-ik·x} of all 10 moments
# (1, Π_i, Π_iΠ_j) vs the deconvolved deposit, using the same resampled particles and the same code as tl_measure.jl.  Light CPU.
#
#   CUDA_VISIBLE_DEVICES="" julia --project=.. -t 8 tl_deposit_check.jl <seed> <sign> <D,D,...> <cfg> [<cfg> ...]
#
# Prints, per config and D, the rms fractional error of each moment over 40 modes (k ≤ 0.12) and, for the density, the bias b in
# (deposit - direct)/direct = -b k².  For a pair of configs, the error of the difference relative to the difference itself.
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
include(joinpath(@__DIR__, "StressTL.jl"))
using .EFTShell, .StressTL, HDF5, FFTW, Printf, Random, JSON3, TOML
FFTW.set_num_threads(1)
scratch() = get(ENV, "EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
const NMESH = parse(Int, get(ENV, "TL_NMESH", "384")); const L = 500.0
const NAMES = ("1", "Πx", "Πy", "Πz", "ΠxΠx", "ΠyΠy", "ΠzΠz", "ΠxΠy", "ΠxΠz", "ΠyΠz")

function modes(kmax, n)
    nm = floor(Int, kmax * L / 2π); v = NTuple{3,Int}[]
    for x in 0:nm, y in -nm:nm, z in -nm:nm
        (x == 0 && (y < 0 || (y == 0 && z <= 0))) && continue
        2π / L * sqrt(x^2 + y^2 + z^2) <= kmax && push!(v, (x, y, z))
    end
    return shuffle(MersenneTwister(1), v)[1:n]
end

function direct(X, V, ms)
    out = zeros(ComplexF64, length(ms), 10); Np = size(X, 1)
    for (m, (nx, ny, nz)) in enumerate(ms)
        kx, ky, kz = 2π / L .* (nx, ny, nz)
        parts = [zeros(ComplexF64, 10) for _ in 1:Threads.nthreads()]
        Threads.@threads :static for t in 1:Threads.nthreads()
            s = parts[t]
            @inbounds for p in (1 + (t - 1) * Np ÷ Threads.nthreads()):(t * Np ÷ Threads.nthreads())
                e = cis(-(kx * X[p, 1] + ky * X[p, 2] + kz * X[p, 3]))
                v1, v2, v3 = V[p, 1], V[p, 2], V[p, 3]
                s[1] += e; s[2] += v1 * e; s[3] += v2 * e; s[4] += v3 * e
                s[5] += v1 * v1 * e; s[6] += v2 * v2 * e; s[7] += v3 * v3 * e; s[8] += v1 * v2 * e; s[9] += v1 * v3 * e; s[10] += v2 * v3 * e
            end
        end
        out[m, :] .= sum(parts) ./ Np
    end
    return out
end

function deposit(X, V, ms)
    M = deposit_moments(X, V, NMESH, L); out = zeros(ComplexF64, length(ms), 10)
    w(i) = i == 0 ? 1.0 : (s = sin(π * i / NMESH) / (π * i / NMESH); s^2)
    for f in 1:10
        F = rfft(view(M, :, :, :, f)) ./ NMESH^3
        for (m, (nx, ny, nz)) in enumerate(ms)
            out[m, f] = F[nx + 1, mod(ny, NMESH) + 1, mod(nz, NMESH) + 1] / (w(nx) * w(ny) * w(nz))
        end
    end
    return out
end

seed = parse(Int, ARGS[1]); sg = parse(Int, ARGS[2]); Ds = parse.(Float64, split(ARGS[3], ",")); cfgs = ARGS[4:end]
ms = modes(0.12, 40); kk = [2π / L * sqrt(sum(abs2, m)) for m in ms]; res = Any[]
for D in Ds
    st = Dict{String,Any}()
    for cfg in cfgs
        f = joinpath(scratch(), cfg, "snap", @sprintf("pm_IR_L%d_S0_s%+d_D%.3f.h5", seed, sg, D))
        Ψ, Π = h5open(h -> (read(h["Psi"]), read(h["Pi"])), f)
        X, V = sheet_resample(Ψ, Π, L; nr=2); Ψ = nothing; Π = nothing
        dep = deposit(X, V, ms); dir = direct(X, V, ms); X = nothing; V = nothing; GC.gc()
        st[cfg] = (dep, dir)
        fr = [sqrt(sum(abs2, dep[:, f] .- dir[:, f]) / sum(abs2, dir[:, f])) for f in 1:10]
        x = -kk .^ 2; r = real.((dep[:, 1] .- dir[:, 1]) .* conj.(dir[:, 1])) ./ abs2.(dir[:, 1]); b = sum(x .* r) / sum(x .^ 2)
        @printf "%s D=%.2f  density rms %.2e, b %+.5f; moments %s\n" cfg D fr[1] b join((@sprintf("%s %.1e", NAMES[f], fr[f]) for f in 2:10), " "); flush(stdout)
        push!(res, Dict("cfg" => cfg, "D" => D, "rms" => fr, "b" => b))
    end
    if length(cfgs) == 2
        (a, A), (c, C) = st[cfgs[1]], st[cfgs[2]]
        pf = [sqrt(sum(abs2, (a[:, f] .- c[:, f]) .- (A[:, f] .- C[:, f])) / sum(abs2, A[:, f] .- C[:, f])) for f in 1:10]
        @printf "pair %s − %s D=%.2f: error of the difference / difference: %s\n" cfgs[1] cfgs[2] D join((@sprintf("%s %.1e", NAMES[f], pf[f]) for f in 1:10), " "); flush(stdout)
        push!(res, Dict("pair" => cfgs, "D" => D, "rms" => pf))
    end
end
open(io -> JSON3.write(io, Dict("modes" => ms, "k" => kk, "seed" => seed, "sign" => sg, "rows" => res)), joinpath(@__DIR__, "..", "results", "tl_deposit_check.json"), "w")
println("CHECK DONE")
