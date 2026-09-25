# Per-element phase-space-sheet products of one displacement snapshot, computed with
# DiscoDJNative's periodic sheet kernels (src/field/sheet_periodic.jl).
#
#   julia -t auto --project=<this directory> sheet_products.jl <psi.npy> <level> <ng> <L> <R1,R2,..> <out.h5>
#
# Output (HDF5, element arrays in C order (i,j,k) of the refined Lagrangian lattice, i.e. the
# same indexing as the numpy snapshot):
#   V        Float64  exact Eulerian element volume (sum of 6 oriented tetrahedra)
#   nflip    Int8     number of inverted tetrahedra of the element
#   nstream  Int32    exact stream count at the element centroid (tetrahedra containing it)
#   rho_c    Float64  AHK sheet density at the centroid (all streams)
#   rhoR_<R> Float64  top-hat (radius R) smoothed sheet density at the centroid, R > 0;
#                     the smoothed field is the FFT top-hat convolution of the point-sampled
#                     sheet density on an ng³ node mesh (sheet_mesh_periodic)
#   attrs    diagnostics (mesh mean density, fraction of even stream counts, timings)
using DiscoDJNative, HDF5, FFTW

include(joinpath(@__DIR__, "npy.jl"))

W_TH(x) = x < 1e-4 ? 1 - x^2 / 10 : 3 * (sin(x) - x * cos(x)) / x^3

function main(args)
    path, level, ng, L = args[1], parse(Int, args[2]), parse(Int, args[3]), parse(Float64, args[4])
    Rs = parse.(Float64, split(args[5], ","; keepempty=false)); out = args[6]
    t0 = time()
    raw = read_npy(path)               # numpy (3,n,n,n) -> Julia (k,j,i,c)
    ψ = permutedims(raw, (3, 2, 1, 4)); raw = nothing   # (i,j,k,c), i = numpy axis 1 (x)
    ψ = refine_displacement(ψ, level)
    n = size(ψ, 1)
    el = sheet_elements_periodic(ψ, L)
    t_el = time() - t0
    pts = reshape(el.centroid, n^3, 3)
    cl = PeriodicCellList(pts, L, L / n)
    q = sheet_query_periodic(ψ, L, pts, cl); cl = nothing
    t_q = time() - t0
    mesh_mean = even_mesh = ms_mesh = NaN
    if !isempty(Rs)                     # the Eulerian mesh is only needed for R > 0
        m = sheet_mesh_periodic(ψ, L, ng)
        mesh_mean = sum(m.density) / ng^3
        even_mesh = count(iseven, m.nstream) / ng^3
        ms_mesh = count(>(1), m.nstream) / ng^3
        m = (density=m.density,)
    end
    ψ = nothing
    t_m = time() - t0
    # C-order (i,j,k) element arrays: Julia (i,j,k) column-major -> write permuted to (k,j,i)
    cperm(a) = permutedims(a, (3, 2, 1))
    h5open(out, "w") do f
        f["V"] = cperm(el.V)
        f["nflip"] = cperm(el.nflip)
        f["nstream"] = cperm(reshape(q.nstream, n, n, n))
        f["rho_c"] = cperm(reshape(q.density, n, n, n))
        if !isempty(Rs)
            ρk = rfft(m.density); m = nothing
            kf = [2π / L * (i <= ng ÷ 2 ? i - 1 : i - 1 - ng) for i in 1:ng]
            kh = [2π / L * (i - 1) for i in 1:(ng ÷ 2 + 1)]
            for R in Rs
                G = similar(ρk)
                Threads.@threads for kk in 1:ng
                    @inbounds for jj in 1:ng, ii in 1:(ng ÷ 2 + 1)
                        G[ii, jj, kk] = ρk[ii, jj, kk] * W_TH(sqrt(kh[ii]^2 + kf[jj]^2 + kf[kk]^2) * R)
                    end
                end
                fR = irfft(G, ng); G = nothing
                v = sample_trilinear_periodic(fR, pts, L); fR = nothing
                f["rhoR_$(R)"] = cperm(reshape(v, n, n, n))
            end
        end
        a = attributes(f)
        a["n"] = n; a["level"] = level; a["ng"] = ng; a["L"] = L; a["R_list"] = Rs
        a["mesh_mean_density"] = mesh_mean
        a["mesh_even_stream_frac"] = even_mesh
        a["mesh_multistream_frac"] = ms_mesh
        a["centroid_even_stream_frac"] = count(iseven, q.nstream) / n^3
        a["centroid_multistream_frac"] = count(>(1), q.nstream) / n^3
        a["flipped_element_frac"] = count(>(0), el.nflip) / n^3
        a["sumV_over_L3_minus1"] = sum(el.V) / L^3 - 1
        a["t_elements_s"] = t_el; a["t_query_s"] = t_q; a["t_mesh_s"] = t_m; a["t_total_s"] = time() - t0
        a["threads"] = Threads.nthreads()
    end
    println("sheet_products $(basename(path)) level=$level n=$n: $(round(time()-t0, digits=1)) s")
end

main(ARGS)
