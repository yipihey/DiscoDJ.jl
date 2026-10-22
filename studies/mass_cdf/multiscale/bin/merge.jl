# Merge the per-case part files into the final CSVs.
include(joinpath(@__DIR__, "_env.jl"))
include(joinpath(@__DIR__, "_csv.jl"))
using DataFrames
resdir = joinpath(M.STUDY_ROOT, "results", "multiscale"); partdir = joinpath(resdir, "parts")
function merged(prefix)
    fs = sort(filter(f -> startswith(f, prefix * "_") && endswith(f, ".csv"), readdir(partdir)))
    isempty(fs) && return DataFrame()
    dfs = [let d = readcsv(joinpath(partdir, f)); d.case = fill(replace(f, prefix * "_" => "", ".csv" => ""), nrow(d)); d; end for f in fs]
    return vcat(dfs...; cols = :union)
end
function reorder(df, first_cols)
    fc = [c for c in first_cols if c in names(df)]
    return df[:, vcat(fc, [c for c in names(df) if !(c in fc)])]
end
loc = merged("locality")
loc = reorder(loc, ["method", "R_f", "L", "N", "h", "eps", "target", "cut", "train_seeds", "test_seeds", "n_cubes_test", "Var_y", "V_C", "V_CP", "R_gain", "R_gain_err",
                    "V_C_train", "V_CP_train", "basis_one", "basis_two", "model"])
M.write_csv(joinpath(resdir, "locality_gain.csv"), loc)
cdf = merged("cdf")
cdf = reorder(cdf, ["method", "R_f", "eps", "cut", "model", "F_M_actual", "F_M_pred_one", "F_M_pred_two", "F_pm_actual", "F_pm_one", "F_pm_two", "F_mp_actual", "F_mp_one", "F_mp_two", "P_dpos"])
M.write_csv(joinpath(resdir, "cdf_multiscale.csv"), cdf)
inn = merged("innov"); isempty(inn) || M.write_csv(joinpath(resdir, "innovation_response.csv"), reorder(inn, ["method", "R_f", "L", "N", "eps", "A_source", "tertile"]))
co = merged("coefs"); isempty(co) || M.write_csv(joinpath(resdir, "oneScale_coefficients.csv"), co)
println("merged: locality $(nrow(loc)) rows, cdf $(nrow(cdf)), innovation $(nrow(inn)), coefs $(nrow(co))")
