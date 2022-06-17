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

T = Float32sr

include("R2.jl")
problems = (eval(problem)(type = Val(T)) for problem ∈ setdiff(names(OptimizationProblems.ADNLPProblems), [:ADNLPProblems]))
my_nlp = problems.f(:woods);


r = R2(my_nlp)

# windows key  + shift + p = open the command then select show profiler 

# ProfileView.@profview R2(my_nlp)
VSCodeServer.@profview R2(my_nlp)

@code_warntype R2(my_nlp)

