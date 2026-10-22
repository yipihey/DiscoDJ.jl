let py = "/Users/tabel/Projects/disco-dj-fem/.venv/bin/python"
    ENV["JULIA_PYTHONCALL_EXE"] = py; ENV["JULIA_CONDAPKG_BACKEND"] = "Null"; get!(ENV, "JAX_PLATFORMS", "cpu")
end
using MassCDF, Statistics, LinearAlgebra, Printf, JLD2, FFTW, StaticArrays
FFTW.set_num_threads(4)
const M = MassCDF
include(joinpath(@__DIR__, "..", "src", "MS.jl"))
