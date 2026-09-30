# Exact phase-space-sheet density on a periodic Cartesian mesh.
#
#   julia -t auto --project=<this directory> sheet_deposit.jl <psi.npy> <ng> <L> <out.npy>
#
# The displacement ψ (numpy (3,N,N,N), x = q + ψ, unwrapped) defines the sheet: every
# Lagrangian cube is split into DiscoDJNative's 6 Kuhn tetrahedra, and each tetrahedron
# carries mass Δq³/6 spread uniformly over its Eulerian volume |V_T| (all streams add; an
# inverted tetrahedron counts with |V_T|).  The mass a tetrahedron puts into a mesh cell is
# its density times the EXACT overlap volume of the tetrahedron with that cell, computed
# by R3D (yipihey/r3djl: r3d clipping + voxelization).  The output is the cell-averaged
# 1 + δ (ρ̄ = 1) in numpy C order [x, y, z], i.e. the same layout as the study's CIC grids.
#
# Mass is conserved to round-off; the only approximation is the sheet itself (linear
# interpolation of the flow between particles).
using DiscoDJNative, R3D

include(joinpath(@__DIR__, "..", "copula", "npy.jl"))

"""    sheet_density_exact(ψ, L, ng) -> (ng,ng,ng) cell-averaged 1+δ

ψ::(n,n,n,3) Julia order (i,j,k,component), unwrapped displacement of lattice point
((i−1)Δq, (j−1)Δq, (k−1)Δq).  Threads over Lagrangian slabs, one grid per task."""
function sheet_density_exact(ψ::AbstractArray{Float64,4}, L::Float64, ng::Int)
    n = size(ψ, 1)
    dq = L / n; h = L / ng
    off = DiscoDJNative._TET_OFFSETS
    sgn = tet_orientation_signs()
    mt = dq^3 / 6                        # tetrahedron mass (ρ̄ = 1 units)
    Vmin = 1e-12 * dq^3                  # below this a tetrahedron is treated as a point mass
    nt = Threads.nthreads()
    chunks = [k for k in 1:n]
    parts = [chunks[c:nt:end] for c in 1:nt]
    grids = Vector{Array{Float64,3}}(undef, nt)
    stats = zeros(Int, nt, 2)            # (point-mass tetrahedra, total tetrahedra)
    @sync for c in 1:nt
        Threads.@spawn begin
            g = zeros(Float64, ng, ng, ng)
            poly = R3D.Flat.FlatPolytope{3,Float64}(64)
            ws = R3D.Flat.VoxelizeWorkspace{3,Float64}(64)
            v = [zeros(3) for _ in 1:4]
            npt = 0; ntot = 0
            for k in parts[c], j in 1:n, i in 1:n, t in 1:6
                for q in 1:4
                    a = off[t, q, 1]; b = off[t, q, 2]; cc = off[t, q, 3]
                    ii = mod1(i + a, n); jj = mod1(j + b, n); kk = mod1(k + cc, n)
                    v[q][1] = (i - 1 + a) * dq + ψ[ii, jj, kk, 1]
                    v[q][2] = (j - 1 + b) * dq + ψ[ii, jj, kk, 2]
                    v[q][3] = (k - 1 + cc) * dq + ψ[ii, jj, kk, 3]
                end
                e1 = v[2] .- v[1]; e2 = v[3] .- v[1]; e3 = v[4] .- v[1]
                V = (e1[1] * (e2[2] * e3[3] - e2[3] * e3[2]) - e1[2] * (e2[1] * e3[3] - e2[3] * e3[1]) +
                     e1[3] * (e2[1] * e3[2] - e2[2] * e3[1])) / 6
                ntot += 1
                if abs(V) < Vmin                     # degenerate: all mass into the cell of its centroid
                    npt += 1
                    cx = (v[1][1] + v[2][1] + v[3][1] + v[4][1]) / 4
                    cy = (v[1][2] + v[2][2] + v[3][2] + v[4][2]) / 4
                    cz = (v[1][3] + v[2][3] + v[3][3] + v[4][3]) / 4
                    g[mod(floor(Int, cx / h), ng) + 1, mod(floor(Int, cy / h), ng) + 1,
                      mod(floor(Int, cz / h), ng) + 1] += mt / h^3
                    continue
                end
                # r3d expects positively oriented tetrahedra
                V > 0 ? R3D.Flat.init_tet!(poly, v[1], v[2], v[3], v[4]) :
                        R3D.Flat.init_tet!(poly, v[1], v[3], v[2], v[4])
                w = mt / abs(V) / h^3                # density contribution per unit overlap volume
                lo, hi = R3D.Flat.get_ibox(poly, (h, h, h))
                R3D.Flat.voxelize_fold!(g, poly, lo, hi, (h, h, h), 0; workspace=ws) do gg, a, b, cc, m
                    @inbounds gg[mod(lo[1] + a - 1, ng) + 1, mod(lo[2] + b - 1, ng) + 1,
                                 mod(lo[3] + cc - 1, ng) + 1] += w * m[1]
                    gg
                end
            end
            grids[c] = g
            stats[c, 1] = npt; stats[c, 2] = ntot
        end
    end
    ρ = grids[1]
    for c in 2:nt
        ρ .+= grids[c]
    end
    return ρ, (point_mass_tets = sum(stats[:, 1]), tets = sum(stats[:, 2]))
end

function main(args)
    path, ng, L, out = args[1], parse(Int, args[2]), parse(Float64, args[3]), args[4]
    t0 = time()
    ψ = permutedims(read_npy(path), (3, 2, 1, 4))          # numpy (3,n,n,n) -> Julia (i,j,k,c)
    ρ, st = sheet_density_exact(ψ, L, ng)
    write_npy(out, Float32.(permutedims(ρ, (3, 2, 1))))    # numpy C order [x,y,z]
    println("sheet_deposit $(basename(path)) ng=$ng: mean=$(sum(ρ) / ng^3) min=$(minimum(ρ)) ",
            "point-mass tets=$(st.point_mass_tets)/$(st.tets) $(round(time() - t0, digits=1)) s")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
