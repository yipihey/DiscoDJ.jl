# Small-scale power-spectrum convergence of the PM runs (single realization, UV member)
#
#   julia --project=. scripts/pk_convergence.jl pkconv_pm256bf pkconv_pm384bf pkconv_pm512lean
#     → results/pk_convergence.h5 (+ printed ratio table)
#
# Density: CIC on a common 512³ mesh (window deconvolved, no shot-noise subtraction; 1/n̄ is stored for reference).
# Displacement: Σ_c |Ψ̂_c|² on each run's particle grid (defined up to that grid's Nyquist frequency).
using HDF5, TOML, Printf
scratch() = get(ENV, "EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
names = ARGS; ref = 2
data = map(names) do n
    d = joinpath(scratch(), n); f = only(filter(x -> startswith(x, "pm_UV") && endswith(x, ".h5"), readdir(d)))
    cfg = TOML.parsefile(joinpath(d, "config.toml"))
    h5open(joinpath(d, f)) do h
        (name=n, k=read(h["pk_k"]), Pd=read(h["pk_dens"]), Pp=read(h["pk_disp"]), D=read(h["epochs"]),
         N=cfg["pm"]["N"], res=cfg["pm"]["res_pm"], L=cfg["L"])
    end
end
h5open(joinpath(@__DIR__, "..", "results", "pk_convergence.h5"), "w") do h
    h["k"] = data[1].k; h["epochs"] = data[1].D
    for d in data
        g = create_group(h, d.name); g["P_dens"] = d.Pd; g["P_disp"] = d.Pp
        attrs(g)["N"] = d.N; attrs(g)["res_pm"] = d.res; attrs(g)["shot_noise"] = (d.L / d.N)^3
        attrs(g)["k_nyq_particles"] = π * d.N / d.L
    end
end
r = data[ref]; k = r.k
for (q, lab) in ((:Pd, "density P(k) (common 512³ deposit)"), (:Pp, "displacement P_Ψ(k) (particle grid)"))
    println("$lab: ratio to $(r.name)")
    for e in findall(D -> D in (0.3, 0.5, 0.7, 1.0), r.D)
        @printf "  D = %.1f   k = %s\n" r.D[e] join([@sprintf("%5.2f", x) for x in k[1:4:end]], "  ")
        for d in data
            d === r && continue
            rat = getfield(d, q)[e, :] ./ getfield(r, q)[e, :]
            @printf "      %-18s %s\n" d.name join([@sprintf("%5.3f", x) for x in rat[1:4:end]], "  ")
        end
    end
end
