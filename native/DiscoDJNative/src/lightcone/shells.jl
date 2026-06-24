"""
Lightcone shell construction.

Provides log-spaced shells in scale factor a, and utilities to map between
scale factor, comoving distance, and shell index — matching DISCO-DJ's
convention exactly.
"""

export log_shells, chi_shells, shell_index_for_a

"""
    log_shells(a_far, a_near, n_shells) -> Vector

Return n_shells+1 shell *edges* log-spaced in a from a_far to a_near.
Convention: a_far < a_near (a_far corresponds to higher redshift / larger χ).
"""
function log_shells(a_far::T, a_near::T, n_shells::Int) where T
    return exp.(LinRange(log(a_far), log(a_near), n_shells + 1))
end

"""
    chi_shells(cosmo, a_edges) -> Vector

Convert a-shell edges to comoving-distance shell edges χ [Mpc/h].
"""
function chi_shells(cosmo::Cosmology, a_edges::AbstractVector)
    return comoving_distance.(Ref(cosmo), a_edges)
end

"""
    shell_index_for_a(a, a_edges) -> Int

Return the shell index k such that a ∈ [a_edges[k], a_edges[k+1]).
Returns -1 if a is outside [a_far, a_near].
"""
function shell_index_for_a(a::Real, a_edges::AbstractVector)
    for k in 1:length(a_edges)-1
        if a_edges[k] <= a < a_edges[k+1]
            return k
        end
    end
    return -1
end
