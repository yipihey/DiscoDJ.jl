# Eulerian mesh products of one displacement snapshot, from DiscoDJNative's periodic sheet kernels.
#
#   julia -t auto --project=<this directory> eulerian_products.jl <psi.npy> <level> <ng> <L> <out.h5>
#
# Output (HDF5; mesh arrays in C order (I,J,K) = numpy [x,y,z] indexing):
#   rho      Float64  point-sampled AHK sheet density at the ng³ nodes, all streams (ρ̄ = 1)
#   nstream  Int8     stream multiplicity at the nodes (saturated at 127)
#   element  Int32    C-order linear index (0-based, numpy ravel order of the (n,n,n) element
#                     array of sheet_products.jl) of the element covering a single-stream node
#                     (−1 where nstream != 1)
using DiscoDJNative, HDF5

include(joinpath(@__DIR__, "npy.jl"))

function main(args)
    path, level, ng, L, out = args[1], parse(Int, args[2]), parse(Int, args[3]), parse(Float64, args[4]), args[5]
    t0 = time()
    ψ = permutedims(read_npy(path), (3, 2, 1, 4))
    ψ = refine_displacement(ψ, level)
    n = size(ψ, 1)
    m = sheet_mesh_periodic(ψ, L, ng)
    loc = sheet_locate_mesh_periodic(ψ, L, ng)
    @assert loc.nstream == m.nstream
    # Julia column-major element index e = i + n(j−1) + n²(k−1)  →  numpy C-order index of [i,j,k]:
    # (i−1)·n² + (j−1)·n + (k−1)
    el = loc.element
    ecc = similar(el)
    Threads.@threads for idx in eachindex(el)
        e = el[idx]
        if loc.nstream[idx] == 1 && e > 0
            e0 = e - 1; i = e0 % n; j = (e0 ÷ n) % n; k = e0 ÷ (n * n)
            ecc[idx] = Int32(i * n * n + j * n + k)
        else
            ecc[idx] = Int32(-1)
        end
    end
    cperm(a) = permutedims(a, (3, 2, 1))
    h5open(out, "w") do f
        f["rho"] = cperm(m.density)
        f["nstream"] = cperm(Int8.(min.(m.nstream, 127)))
        f["element"] = cperm(ecc)
        a = attributes(f)
        a["n"] = n; a["level"] = level; a["ng"] = ng; a["L"] = L
        a["mean_rho"] = sum(m.density) / ng^3
        a["multistream_volume_frac"] = count(>(1), m.nstream) / ng^3
        a["even_stream_frac"] = count(iseven, m.nstream) / ng^3
        a["t_total_s"] = time() - t0
    end
    println("eulerian_products $(basename(path)) level=$level n=$n ng=$ng: $(round(time() - t0, digits=1)) s")
end

main(ARGS)
