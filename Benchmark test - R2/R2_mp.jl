export bmark_solvers

function bmark_solvers(solver,problem_suit::Dict{Symbol, <:Any}, args...; kwargs...)
    stats = Dict{Symbol, DataFrame}()
    for (name, problems) in problem_suit
        @debug "running" name
        stats[name] = solve_problems(solver, problems, args ...; kwargs...)
    end
    return stats
end

using JSOSolvers
using LinearAlgebra
using Random
using Printf
using DataFrames
using OptimizationProblems
using NLPModels
using ADNLPModels
using OptimizationProblems.ADNLPProblems
using SolverCore
using SolverBenchmark
using DoubleFloats
using Quadmath
using BFloat16s
using Statistics
using CSV
using Plots
using Profile
using ProfileView
using StochasticRounding


include("R2.jl")
include("R2_D_M.jl")

problemsF32 = (eval(problem)(type = Val(Float32)) for problem ∈ setdiff(names(OptimizationProblems.ADNLPProblems), [:ADNLPProblems]))
problemsF32sr = (eval(problem)(type = Val(Float32sr)) for problem ∈ setdiff(names(OptimizationProblems.ADNLPProblems), [:ADNLPProblems]))

solver = Dict(
            :R2 => 
              nlp -> R2(
                  nlp;
                  maxiterations=250,
                  η1 = 1e-4,
                  η2 = 0.75,
                  γ1 = 0.33,
                  γ2 = 2.0,
                  verbose = false)
                )       

statsF32 = bmark_solvers(solver, problemsF32,  skipif = prob -> (!unconstrained(prob) || get_nvar(prob) > 100 || get_nvar(prob) < 5))
