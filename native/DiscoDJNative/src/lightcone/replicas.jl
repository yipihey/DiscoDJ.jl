"""
Periodic-box replica enumeration for the past lightcone.

Finds all periodic replicas r·L such that the corresponding AABB
[observer - χ_far, observer + χ_far]³ intersects the sphere of radius χ_far
(and is outside χ_near after wrapping).

Matches DISCO-DJ's `enumerate_replicas(L, observer, chi_near, chi_far)`.
"""

export enumerate_replicas

using LinearAlgebra

"""
    enumerate_replicas(boxsize, observer, chi_near, chi_far) -> Matrix{Int}

Return an (n_replicas, 3) integer matrix of replica offsets.
Each row r gives the offset vector r·L to be added to particle positions.
Only replicas whose AABB shell [chi_near, chi_far] intersects the observer
sphere are included.

`chi_near`, `chi_far` in Mpc/h (chi_near < chi_far; chi_near for a_near close=1).
"""
function enumerate_replicas(boxsize::Real, observer::AbstractVector,
                            chi_near::Real, chi_far::Real)
    L = boxsize
    # Maximum replica index needed in each direction
    r_max = ceil(Int, (chi_far + maximum(abs.(observer .- L/2))) / L) + 1

    replicas = Vector{NTuple{3,Int}}()
    for rx in -r_max:r_max, ry in -r_max:r_max, rz in -r_max:r_max
        # Box corner nearest to observer in replica frame
        box_origin = [rx*L, ry*L, rz*L]
        box_far    = box_origin .+ L

        # Closest and farthest point in box from observer
        nearest  = clamp.(observer, box_origin, box_far)
        d_near   = norm(nearest  .- observer)
        # Farthest corner
        farthest = [observer[d] < box_origin[d] + L/2 ? box_far[d] : box_origin[d] for d in 1:3]
        d_far    = norm(farthest .- observer)

        # Replica contributes if it overlaps the shell [chi_near, chi_far]
        if d_near <= chi_far && d_far >= chi_near
            push!(replicas, (rx, ry, rz))
        end
    end

    # Convert to matrix
    n = length(replicas)
    out = Matrix{Int}(undef, n, 3)
    for (i, (rx, ry, rz)) in enumerate(replicas)
        out[i, 1] = rx; out[i, 2] = ry; out[i, 3] = rz
    end
    return out
end
