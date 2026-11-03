# ── Experiment 3: the stress operator basis of arXiv:2509.05187 at field level (CPU) ───────────────────────────────
#
# From the realization's linear field δ̃^(1) (D = 1) filtered with W_Λ, build the SPT fields δ̃^(1..3), θ̃^(2) by the
# F_n/G_n recursion (Eqs. 2.34–2.36) in configuration space,
#   δ_n = Σ_m [(2n+1) A(θ_m, δ_{n−m}) + B(θ_m, θ_{n−m})] / ((2n+3)(n−1)),
#   θ_n = Σ_m [3 A(θ_m, δ_{n−m}) + n B(θ_m, θ_{n−m})] / ((2n+3)(n−1)),
#   A(θ, δ) = ∇·(δ ∇∇⁻²θ),  B(θ, θ') = ∇²(∇∇⁻²θ · ∇∇⁻²θ'),
# and every stress function C^{ij(n)}_{O,α} of App. A.2, Eq. (A.20), plus δ^K_ij ∂²δ̃^(1) (the paper's C_{∂²δ,1}·k_NL²).
# Each function is returned on the `TLModes` as (M, 6) Fourier coefficients (component order `COMP`).
module OpsTL

using FFTW, LinearAlgebra
using ..StressTL: TLModes, extract, khalf, kvec1, COMP

export OPS, build_operators, spt_fields

# (name, order n, coefficient label) — the coefficient multiplies Σ_n D^{n+2} s_L^n C^{(n)} (Eq. 2.50)
const OPS = [
    ("C1_r1", 1, "r1"), ("C1_d1", 1, "d1"),
    ("C2_r1", 2, "r1"), ("C2_r2", 2, "r2"), ("C2_d1", 2, "d1"), ("C2_d2", 2, "d2"),
    ("C2_r2_1", 2, "rr1"), ("C2_rd1", 2, "rd1"), ("C2_dd1", 2, "dd1"),
    ("C3_r1", 3, "r1"), ("C3_r2", 3, "r2"), ("C3_r3", 3, "r3"), ("C3_d1", 3, "d1"), ("C3_d2", 3, "d2"), ("C3_d3", 3, "d3"),
    ("C3_rr1", 3, "rr1"), ("C3_rr2", 3, "rr2"), ("C3_rd1", 3, "rd1"), ("C3_rd2", 3, "rd2"), ("C3_dd1", 3, "dd1"),
    ("C3_dd2", 3, "dd2"), ("C3_rrr1", 3, "rrr1"), ("C3_rrd1", 3, "rrd1"), ("C3_rdd1", 3, "rdd1"), ("C3_ddd1", 3, "ddd1"),
    ("C3_trr2r1", 3, "trr2r1"), ("Cd2_d1", 1, "lapd1")]
# coefficient labels ↔ paper: r1=c_{r,1}, r2=c_{r,2}, r3=c_{r,3}, d1..d3=c_{δ,1..3}, rr1/rr2=c_{r²,1/2}, rd1/rd2=c_{rδ,1/2},
# dd1/dd2=c_{δ²,1/2}, rrr1=c_{r³,1}, rrd1=c_{r²δ,1}, rdd1=c_{rδ²,1}, ddd1=c_{δ³,1}, trr2r1=c_{(r²)r,1}, lapd1=c_{∂²δ,1}/k_NL²
# ("Cd2_d1" is D⁵ like an n = 3 term but linear in δ: its D power is set explicitly in the regression)

struct Grid
    N::Int; L::Float64
    k::NTuple{3,Array{Float64,3}}; k2::Array{Float64,3}
end
function Grid(N, L)
    kx = khalf(N, L); ky = kvec1(N, L); kz = kvec1(N, L)
    k = (reshape(kx, :, 1, 1) .+ 0 .* reshape(ky, 1, :, 1) .+ 0 .* reshape(kz, 1, 1, :),
         0 .* reshape(kx, :, 1, 1) .+ reshape(ky, 1, :, 1) .+ 0 .* reshape(kz, 1, 1, :),
         0 .* reshape(kx, :, 1, 1) .+ 0 .* reshape(ky, 1, :, 1) .+ reshape(kz, 1, 1, :))
    k2 = k[1] .^ 2 .+ k[2] .^ 2 .+ k[3] .^ 2
    return Grid(N, Float64(L), k, k2)
end
ir(g::Grid, fh) = irfft(fh, g.N)
invlap(g::Grid, fh) = (o = -fh ./ max.(g.k2, 1e-300); o[1] = 0; o)        # ∂⁻²
# real-space ∂_{i1}…∂_{im} ∂⁻² f  (m indices)
dpot(g::Grid, fh, idx...) = (o = invlap(g, fh); for i in idx; o = im .* g.k[i] .* o; end; ir(g, o))
deriv(g::Grid, fh, idx...) = (o = copy(fh); for i in idx; o = im .* g.k[i] .* o; end; ir(g, o))

A(g, θh, δ) = (φ = invlap(g, θh); out = zeros(ComplexF64, size(θh));
               for i in 1:3; out .+= im .* g.k[i] .* rfft(δ .* ir(g, im .* g.k[i] .* φ)); end; out)
B(g, θh1, θh2) = (φ1 = invlap(g, θh1); φ2 = invlap(g, θh2); s = zeros(g.N, g.N, g.N);
                  for i in 1:3; s .+= ir(g, im .* g.k[i] .* φ1) .* ir(g, im .* g.k[i] .* φ2); end; -g.k2 .* rfft(s))

"""SPT δ̂_n, θ̂_n (n = 1..nmax) of the linear field δ̂₁ (rfft on grid g)."""
function spt_fields(g::Grid, δ1h; nmax::Int=3)
    δ = Dict(1 => δ1h); θ = Dict(1 => δ1h); δr = Dict(1 => ir(g, δ1h))
    for n in 2:nmax
        dn = zeros(ComplexF64, size(δ1h)); tn = similar(dn); tn .= 0
        for m in 1:n-1
            a = A(g, θ[m], δr[n - m]); b = B(g, θ[m], θ[n - m])
            dn .+= (2n + 1) .* a .+ b; tn .+= 3 .* a .+ n .* b
        end
        δ[n] = dn ./ ((2n + 3) * (n - 1)); θ[n] = tn ./ ((2n + 3) * (n - 1)); δr[n] = ir(g, δ[n])
    end
    return δ, θ
end

oversamp(filter) = parse(Float64, get(ENV, "TL_OPS_KNY", filter === :sharp ? "3" : "7"))

"""
    build_operators(δLh, gN, L, Λ, md) -> Dict(name => (M,6) modes)

δLh: rfft (x-half layout, grid gN) of the linear δ at D = 1.  Filters with W_Λ (cut at 3.5Λ), builds the SPT fields and
all C-functions on a grid whose Nyquist is ≥ 7Λ (alias-free cubic products), and extracts them on `md`."""
function build_operators(δL::Array{Float64,3}, L::Real, Λ::Real, md::TLModes; filter::Symbol=:sharp, responses::Bool=true)
    Nf = size(δL, 1)
    # alias-free for the fitted modes (k < Λ): all fields are at most cubic in δ₁, so with sharp input (≤ Λ) a Nyquist of 3Λ
    # suffices (2k_Ny − 3Λ > Λ, and second-order content ≤ 2Λ is represented exactly); Gaussian tails need 7Λ.
    No = min(Nf, max(64, 2 * ceil(Int, oversamp(filter) * Λ * L / (2π) + 1)))
    gf = Grid(Nf, L); g = Grid(No, L)
    Fh = rfft(δL)
    kkeep = filter === :sharp ? Λ : 3.5Λ
    δ1h = zeros(ComplexF64, No ÷ 2 + 1, No, No)                    # filtered, transferred to the operator grid
    h = No ÷ 2
    for l in 1:No, j in 1:No, i in 1:h
        my = j - 1 < h ? j - 1 : j - 1 - No; mz = l - 1 < h ? l - 1 : l - 1 - No
        (my == -h || mz == -h) && continue
        k2 = (2π / L)^2 * ((i - 1)^2 + my^2 + mz^2)
        k2 >= kkeep^2 && continue
        δ1h[i, j, l] = Fh[i, mod(my, Nf) + 1, mod(mz, Nf) + 1] * (filter === :sharp ? 1.0 : exp(-k2 / (2Λ^2))) * (No / Nf)^3
    end
    δ, θ = spt_fields(g, δ1h; nmax=3)
    d1 = ir(g, δ[1]); d2 = ir(g, δ[2]); d3 = ir(g, δ[3])
    s = [dpot(g, δ[1], i) for i in 1:3]                                 # ∂_k δ₁/∂²
    r = Dict((i, j) => dpot(g, δ[1], i, j) for i in 1:3, j in 1:3)     # r_ij
    t = Dict((i, j, k) => dpot(g, δ[1], i, j, k) for i in 1:3, j in 1:3, k in 1:3)
    gd1 = [deriv(g, δ[1], i) for i in 1:3]                              # ∂_k δ₁
    gd2 = [deriv(g, δ[2], i) for i in 1:3]
    χ2 = [dpot(g, θ[2], i) for i in 1:3]                                # ∂_k θ₂/∂²
    r2 = Dict((i, j) => dpot(g, δ[2], i, j) for i in 1:3, j in 1:3)    # ∂_i∂_j δ₂/∂²
    t2s = Dict((i, j) => sum(dpot(g, δ[2], i, j, k) .* s[k] for k in 1:3) for (i, j) in COMP)   # ∂_i∂_j∂_kδ₂/∂² s_k
    r3 = Dict((i, j) => dpot(g, δ[3], i, j) for (i, j) in COMP)
    ts = Dict((i, j) => sum(t[(i, j, k)] .* s[k] for k in 1:3) for (i, j) in COMP)               # t_ijk s_k
    tχ = Dict((i, j) => sum(t[(i, j, k)] .* χ2[k] for k in 1:3) for (i, j) in COMP)
    uss = Dict((i, j) => sum(dpot(g, δ[1], i, j, m, n) .* s[m] .* s[n] for m in 1:3, n in 1:3) for (i, j) in COMP)
    rs = [sum(r[(m, n)] .* s[n] for n in 1:3) for m in 1:3]            # r_mn s_n
    trs = Dict((i, j) => sum(t[(m, i, j)] .* rs[m] for m in 1:3) for (i, j) in COMP)
    gs1 = sum(gd1[k] .* s[k] for k in 1:3)                              # ∂_kδ₁ s_k
    gs2 = sum(gd2[k] .* s[k] for k in 1:3)
    gχ = sum(gd1[k] .* χ2[k] for k in 1:3)
    hss = sum(deriv(g, δ[1], m, n) .* s[m] .* s[n] for m in 1:3, n in 1:3)   # ∂_m∂_nδ₁ s_m s_n
    grs = sum(gd1[m] .* rs[m] for m in 1:3)                             # ∂_mδ₁ r_mn s_n
    rr = Dict((i, j) => sum(r[(i, k)] .* r[(k, j)] for k in 1:3) for (i, j) in COMP)
    trr = sum(r[(m, n)] .^ 2 for m in 1:3, n in 1:3)
    rrr = Dict((i, j) => sum(r[(i, m)] .* r[(m, n)] .* r[(n, j)] for m in 1:3, n in 1:3) for (i, j) in COMP)
    rtsym = Dict((i, j) => sum(r[(i, m)] .* t[(n, m, j)] .* s[n] .+ r[(j, m)] .* t[(n, m, i)] .* s[n] for m in 1:3, n in 1:3)
                 for (i, j) in COMP)
    r2r = Dict((i, j) => sum(r2[(i, k)] .* r[(k, j)] .+ r2[(j, k)] .* r[(k, i)] for k in 1:3) for (i, j) in COMP)
    lap1 = ir(g, -g.k2 .* δ[1])
    δK(i, j) = i == j ? 1.0 : 0.0
    F = Dict{String,Function}(
        "C1_r1" => (i, j) -> r[(i, j)],                    "C1_d1" => (i, j) -> δK(i, j) .* d1,
        "C2_r1" => (i, j) -> ts[(i, j)],                   "C2_d1" => (i, j) -> δK(i, j) .* gs1,
        "C2_r2" => (i, j) -> r2[(i, j)] .- ts[(i, j)],     "C2_d2" => (i, j) -> δK(i, j) .* (d2 .- gs1),
        "C2_rd1" => (i, j) -> r[(i, j)] .* d1,              "C2_r2_1" => (i, j) -> rr[(i, j)],
        "C2_dd1" => (i, j) -> δK(i, j) .* d1 .^ 2,
        "C3_r1" => (i, j) -> 0.5 .* (tχ[(i, j)] .+ uss[(i, j)] .+ trs[(i, j)]),
        "C3_r2" => (i, j) -> t2s[(i, j)] .- uss[(i, j)] .- trs[(i, j)],
        "C3_r3" => (i, j) -> r3[(i, j)] .- t2s[(i, j)] .- 0.5 .* tχ[(i, j)] .+ 0.5 .* uss[(i, j)] .+ 0.5 .* trs[(i, j)],
        "C3_d1" => (i, j) -> δK(i, j) .* 0.5 .* (gχ .+ hss .+ grs),
        "C3_d2" => (i, j) -> δK(i, j) .* (gs2 .- hss .- grs),
        "C3_d3" => (i, j) -> δK(i, j) .* 0.5 .* (2 .* d3 .- 2 .* gs2 .- gχ .+ hss .+ grs),
        "C3_rr1" => (i, j) -> rtsym[(i, j)],
        "C3_rr2" => (i, j) -> r2r[(i, j)] .- rtsym[(i, j)],
        "C3_rd1" => (i, j) -> d1 .* ts[(i, j)] .+ r[(i, j)] .* gs1,
        "C3_rd2" => (i, j) -> r2[(i, j)] .* d1 .+ r[(i, j)] .* d2 .- ts[(i, j)] .* d1 .- r[(i, j)] .* gs1,
        "C3_dd1" => (i, j) -> δK(i, j) .* 2 .* d1 .* gs1,
        "C3_dd2" => (i, j) -> δK(i, j) .* 2 .* (d2 .* d1 .- d1 .* gs1),
        "C3_rrr1" => (i, j) -> rrr[(i, j)],                 "C3_rrd1" => (i, j) -> d1 .* rr[(i, j)],
        "C3_rdd1" => (i, j) -> d1 .^ 2 .* r[(i, j)],        "C3_ddd1" => (i, j) -> δK(i, j) .* d1 .^ 3,
        "C3_trr2r1" => (i, j) -> trr .* r[(i, j)],
        "Cd2_d1" => (i, j) -> δK(i, j) .* lap1)
    out = Dict{String,Matrix{ComplexF64}}()
    z = zeros(No, No, No)
    for (name, _, _) in OPS
        out[name] = reduce(hcat, [extract(rfft(F[name](i, j) .+ z), md, No) for (i, j) in COMP])
    end
    resp = Dict{String,Any}()
    if responses
        # counterterm responses (Eqs. 2.57–2.62), per coefficient label with unit coefficient, time-independent parts:
        # δ_ct = D³δ̃¹ + D⁴δ̃² + D⁵δ̃³ (+ D⁵ δ̃_∂²),  π̃_ct,(1), π̃_ct,(2)  (k_NL factors absorbed into c̄)
        π1 = s; π2 = [χ2[c] .+ d1 .* s[c] for c in 1:3]                 # π̃_(1), π̃_(2) = [(1+δ)∂θ/∂²]^(n)
        δ2pot = [dpot(g, δ[2], c) for c in 1:3]                           # ∂_kδ̃₂/∂²
        R(fh) = ir(g, fh)
        ddiv(T) = sum((i == j ? 1.0 : 2.0) .* (-(g.k[i] .* g.k[j])) .* rfft(T[(i, j)]) for (i, j) in COMP)     # ∂_i∂_jT_ij (T sym)
        full6(T) = Dict((i, j) => (i <= j ? T[(i, j)] : T[(j, i)]) for i in 1:3, j in 1:3)
        divv(T) = [R(sum(im .* g.k[j] .* rfft(T[(i, j)]) for j in 1:3)) for i in 1:3]                         # ∂_jT_ij
        labels = unique(lab for (_, _, lab) in OPS)
        for lab in labels
            τ = Dict(n => Dict((i, j) => zeros(No, No, No) for (i, j) in COMP) for n in 1:3)
            for (name, n, l) in OPS
                (l == lab && name != "Cd2_d1") || continue
                for (i, j) in COMP; τ[n][(i, j)] .+= F[name](i, j) .+ z; end
            end
            if lab == "lapd1"   # δ_∂²ct = (1/26) ∂_i∂_j(δ_ij ∂²δ̃₁) → D⁵, linear
                h1 = (g.k2 .^ 2) .* δ[1] ./ 26
                resp[lab] = (d1=zeros(ComplexF64, size(md.nvec, 1)), d2=zeros(ComplexF64, size(md.nvec, 1)),
                             d3=extract(h1, md, No), p1=zeros(ComplexF64, size(md.nvec, 1), 3), p2=zeros(ComplexF64, size(md.nvec, 1), 3))
                continue
            end
            T1 = full6(τ[1]); T2 = full6(τ[2]); T3 = full6(τ[3])
            dc1h = ddiv(τ[1]) ./ 9
            pc1 = [R(sum(im .* g.k[a] .* g.k[j] .* g.k[kk] ./ max.(g.k2, 1e-300) .* rfft(T1[(j, kk)])
                         for j in 1:3, kk in 1:3)) ./ 3 for a in 1:3]                           # (1/3) ∂_i∂_j∂_k/∂² τ̃₁_jk
            dT1 = divv(T1)                                                                      # ∂_kτ̃₁_jk
            M2 = Dict((i, j) => T2[(i, j)] .+ s[i] .* dT1[j] .- (i == j ? sum(s[k] .* dT1[k] for k in 1:3) ./ 6 : 0.0) for i in 1:3, j in 1:3)
            dc2h = sum(-(g.k[i] .* g.k[j]) .* rfft(M2[(i, j)]) for i in 1:3, j in 1:3) .* (2 / 33)
            A = Dict((j, k) => T2[(j, k)] .+ s[j] .* dT1[k] for j in 1:3, k in 1:3)
            sdT = sum(s[j] .* dT1[j] for j in 1:3)
            dT2 = divv(T2)
            pc2 = [(2 / 9) .* (dT2[i] .+ (1 / 11) .* R(sum(im .* g.k[i] .* g.k[j] .* g.k[k] ./ max.(g.k2, 1e-300) .* rfft(A[(j, k)])
                                                           for j in 1:3, k in 1:3))      # ∂_i∂_j∂_k/∂² ↔ i k_ik_jk_k/k²
                                .- (2 / 11) .* R(im .* g.k[i] .* rfft(sdT))
                                .+ 0.5 .* R(sum(im .* g.k[l] .* rfft(s[i] .* dT1[l] .+ s[l] .* dT1[i]) for l in 1:3))) for i in 1:3]
            dc1 = R(dc1h); dc2 = R(dc2h)
            p_dc1 = [dpot(g, dc1h, c) for c in 1:3]; p_dc2 = [dpot(g, dc2h, c) for c in 1:3]
            M3 = Dict((i, j) => 3 .* (p_dc2[i] .* s[j] .+ p_dc1[i] .* δ2pot[j])
                                .- (i == j ? 1.5 .* sum(p_dc2[k] .* s[k] .+ p_dc1[k] .* δ2pot[k] for k in 1:3) : 0.0)
                                .+ 2 .* (pc2[i] .* π1[j] .+ pc1[i] .* π2[j]) .- 2 .* pc1[i] .* π1[j] .* d1 .- π1[i] .* π1[j] .* dc1
                                .+ T3[(i, j)] for i in 1:3, j in 1:3)
            dc3h = sum(-(g.k[i] .* g.k[j]) .* rfft(M3[(i, j)]) for i in 1:3, j in 1:3) ./ 26
            resp[lab] = (d1=extract(dc1h, md, No), d2=extract(dc2h, md, No), d3=extract(dc3h, md, No),
                         p1=reduce(hcat, [extract(rfft(pc1[c]), md, No) for c in 1:3]),
                         p2=reduce(hcat, [extract(rfft(pc2[c]), md, No) for c in 1:3]))
        end
    end
    return out, Dict(n => extract(δ[n], md, No) for n in 1:3), Dict(n => extract(θ[n], md, No) for n in 1:3), resp
end

end # module
