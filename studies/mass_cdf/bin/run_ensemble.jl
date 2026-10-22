# Worker: run a slice of seeds of one ensemble configuration.
#   julia -t 4 --project=studies/mass_cdf studies/mass_cdf/bin/run_ensemble.jl <config.toml> <ic_type> <seed_lo> <seed_hi> [outdir]
# Python interpreter: DISCO-DJ's venv (set before PythonCall loads, see DiscoDJLib docs).
let py = get(ENV, "DISCODJ_PYTHON", normpath(joinpath(@__DIR__, "..", "..", "..", "..", "disco-dj-fem", ".venv", "bin", "python")))
    get!(ENV, "JULIA_PYTHONCALL_EXE", py); ENV["JULIA_CONDAPKG_BACKEND"] = "Null"; get!(ENV, "JAX_PLATFORMS", "cpu")
end
using MassCDF
cfgpath, ic_type, lo, hi = ARGS[1], ARGS[2], parse(Int, ARGS[3]), parse(Int, ARGS[4])
cfg = load_config(cfgpath)
outdir = length(ARGS) >= 5 ? ARGS[5] : joinpath(MassCDF.STUDY_ROOT, "data", cfg["name"])
mkpath(outdir)
out = joinpath(outdir, "part_$(ic_type)_$(lo)-$(hi).jld2")
logf = open(replace(out, ".jld2" => ".log"), "a")
run_ensemble(cfg; seeds = lo:hi, ic_type, out, log = logf)
close(logf)
