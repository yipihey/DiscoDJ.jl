# Worker: FEM N-body vs nLPT on identical phases.
#   julia -t 4 --project=studies/mass_cdf studies/mass_cdf/bin/run_fem.jl <config.toml> <seed_lo> <seed_hi> [outdir]
let py = get(ENV, "DISCODJ_PYTHON", normpath(joinpath(@__DIR__, "..", "..", "..", "..", "disco-dj-fem", ".venv", "bin", "python")))
    get!(ENV, "JULIA_PYTHONCALL_EXE", py); ENV["JULIA_CONDAPKG_BACKEND"] = "Null"; get!(ENV, "JAX_PLATFORMS", "cpu")
end
using MassCDF
cfgpath, lo, hi = ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3])
cfg = load_config(cfgpath)
outdir = length(ARGS) >= 4 ? ARGS[4] : joinpath(MassCDF.STUDY_ROOT, "data", cfg["name"])
mkpath(outdir)
out = joinpath(outdir, "part_fem_$(lo)-$(hi).jld2")
logf = open(replace(out, ".jld2" => ".log"), "a")
run_fem_ensemble(cfg; seeds = lo:hi, out, log = logf)
close(logf)
