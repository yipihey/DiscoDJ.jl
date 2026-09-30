# Mass-weighted density PDFs straight from the phase-space sheet, without gridding the weights.
#
#   julia -t auto --project=<this directory> tet_mass_pdf.jl <psi.npy> <L> <out.json> [R1=field1.npy R2=field2.npy ...]
#
# Every tetrahedron (6 Kuhn tetrahedra per Lagrangian cube, DiscoDJNative's tessellation) has the
# same mass Δq³/6, so a sample over tetrahedra with equal weights IS a mass-weighted sample:
#   R = 0 ("tet")  its stream density ρ_T = (Δq³/6)/|V_T| (ρ̄ = 1): the unsmoothed sheet density
#                  carried by that mass element (per stream where the flow is multi-stream)
#   R > 0          the smoothed field ρ_R (numpy [x,y,z] float32 on an ng³ mesh whose values sit
#                  at cell centres (i+½)h, e.g. the smoothed exact sheet density) interpolated
#                  trilinearly (periodic) to the tetrahedron centroid
# Output JSON per R: quantiles at QS, the fine CDF of log10 ρ on FINEBINS and the PDF on LOGBINS
# (the binning of study 1), and the fraction of inverted tetrahedra.
using DiscoDJNative

include(joinpath(@__DIR__, "..", "copula", "npy.jl"))

const QS = [1e-3, 1e-2, 0.1, 0.5, 0.9, 0.99, 0.999]
const FINEBINS = range(log10(0.05), log10(20.0), length=1201)
const LOGBINS = range(log10(0.05), log10(20.0), length=161)

@inline function trilinear(f, x, y, z, h, ng)
    fx = x / h - 0.5; fy = y / h - 0.5; fz = z / h - 0.5          # cell-centred samples
    ix = floor(Int, fx); iy = floor(Int, fy); iz = floor(Int, fz)
    dx = fx - ix; dy = fy - iy; dz = fz - iz
    i0 = mod(ix, ng) + 1; j0 = mod(iy, ng) + 1; k0 = mod(iz, ng) + 1
    i1 = mod(ix + 1, ng) + 1; j1 = mod(iy + 1, ng) + 1; k1 = mod(iz + 1, ng) + 1
    @inbounds return (f[i0, j0, k0] * (1 - dx) * (1 - dy) * (1 - dz) + f[i1, j0, k0] * dx * (1 - dy) * (1 - dz) +
                      f[i0, j1, k0] * (1 - dx) * dy * (1 - dz) + f[i0, j0, k1] * (1 - dx) * (1 - dy) * dz +
                      f[i1, j1, k0] * dx * dy * (1 - dz) + f[i1, j0, k1] * dx * (1 - dy) * dz +
                      f[i0, j1, k1] * (1 - dx) * dy * dz + f[i1, j1, k1] * dx * dy * dz)
end

"""    tet_samples(ψ, L, fields) -> (vals::Vector{Vector{Float32}}, nflip)

vals[1] = stream density of every tetrahedron, vals[1+r] = fields[r] at every centroid."""
function tet_samples(ψ::AbstractArray{Float64,4}, L::Float64, fields::Vector{<:AbstractArray{Float32,3}})
    n = size(ψ, 1); dq = L / n; mt = dq^3 / 6
    off = DiscoDJNative._TET_OFFSETS
    nT = 6 * n^3
    vals = [Vector{Float32}(undef, nT) for _ in 1:(1 + length(fields))]
    nflip = zeros(Int, n)
    Threads.@threads for k in 1:n
        nf = 0
        @inbounds for j in 1:n, i in 1:n, t in 1:6
            idx = (((k - 1) * n + (j - 1)) * n + (i - 1)) * 6 + t
            vx = ntuple(q -> (i - 1 + off[t, q, 1]) * dq + ψ[mod1(i + off[t, q, 1], n), mod1(j + off[t, q, 2], n), mod1(k + off[t, q, 3], n), 1], 4)
            vy = ntuple(q -> (j - 1 + off[t, q, 2]) * dq + ψ[mod1(i + off[t, q, 1], n), mod1(j + off[t, q, 2], n), mod1(k + off[t, q, 3], n), 2], 4)
            vz = ntuple(q -> (k - 1 + off[t, q, 3]) * dq + ψ[mod1(i + off[t, q, 1], n), mod1(j + off[t, q, 2], n), mod1(k + off[t, q, 3], n), 3], 4)
            e1 = (vx[2] - vx[1], vy[2] - vy[1], vz[2] - vz[1])
            e2 = (vx[3] - vx[1], vy[3] - vy[1], vz[3] - vz[1])
            e3 = (vx[4] - vx[1], vy[4] - vy[1], vz[4] - vz[1])
            V = (e1[1] * (e2[2] * e3[3] - e2[3] * e3[2]) - e1[2] * (e2[1] * e3[3] - e2[3] * e3[1]) +
                 e1[3] * (e2[1] * e3[2] - e2[2] * e3[1])) / 6
            vals[1][idx] = Float32(mt / max(abs(V), 1e-300))
            cx = mod(sum(vx) / 4, L); cy = mod(sum(vy) / 4, L); cz = mod(sum(vz) / 4, L)
            for r in eachindex(fields)
                f = fields[r]
                vals[1 + r][idx] = Float32(trilinear(f, cx, cy, cz, L / size(f, 1), size(f, 1)))
            end
            # orientation relative to the undeformed lattice (DiscoDJNative convention)
            V * tet_orientation_signs()[t] <= 0 && (nf += 1)
        end
        nflip[k] = nf
    end
    return vals, sum(nflip) / nT
end

function summarize(v::Vector{Float32})
    sort!(v)
    nv = length(v)
    q = [begin
             p = qq * nv + 0.5                      # (rank − ½)/n convention of wquantile
             lo = clamp(floor(Int, p), 1, nv); hi = clamp(lo + 1, 1, nv); w = clamp(p - lo, 0, 1)
             Float64(v[lo]) * (1 - w) + Float64(v[hi]) * w
         end for qq in QS]
    lv = log10.(max.(v, 1f-6))
    cnt(b) = [searchsortedfirst(lv, b[i + 1]) - searchsortedfirst(lv, b[i]) for i in 1:length(b)-1]
    hf = cnt(FINEBINS); hl = cnt(LOGBINS)
    cdf = cumsum(hf) ./ sum(hf)
    pdf = hl ./ (nv * step(LOGBINS))
    return Dict("qM" => q, "cdfM" => cdf, "pdfM" => pdf, "min" => Float64(v[1]), "max" => Float64(v[end]))
end

function main(args)
    path, L, out = args[1], parse(Float64, args[2]), args[3]
    Rs = Float64[]; fields = Array{Float32,3}[]
    for a in args[4:end]
        R, f = split(a, "=")
        push!(Rs, parse(Float64, R)); push!(fields, Float32.(permutedims(read_npy(f), (3, 2, 1))))
    end
    t0 = time()
    ψ = permutedims(read_npy(path), (3, 2, 1, 4))
    vals, fflip = tet_samples(ψ, L, fields)
    ψ = nothing; fields = nothing
    res = Dict{String,Any}("flipped_tet_fraction" => fflip, "n_tets" => length(vals[1]))
    res["R0"] = summarize(vals[1]); vals[1] = Float32[]
    for (r, R) in enumerate(Rs)
        res["R$(R == round(R) ? Int(R) : R)"] = summarize(vals[1 + r]); vals[1 + r] = Float32[]
    end
    open(out, "w") do io
        # minimal JSON writer (no extra dependency)
        function w(x)
            if x isa Dict
                print(io, "{"); first = true
                for (k, v) in x
                    first || print(io, ","); first = false
                    print(io, "\"", k, "\":"); w(v)
                end
                print(io, "}")
            elseif x isa AbstractVector
                print(io, "["); for (i, v) in enumerate(x); i > 1 && print(io, ","); w(v); end; print(io, "]")
            else
                print(io, isfinite(x) ? x : 0)
            end
        end
        w(res)
    end
    println("tet_mass_pdf $(basename(path)): $(length(Rs)) smoothed fields, flipped tets $(round(fflip, sigdigits=3)), $(round(time() - t0, digits=1)) s")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
