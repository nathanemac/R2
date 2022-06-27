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

test = (eval(problem)(type = Val(Float32sr)) for problem ∈ setdiff(names(OptimizationProblems.ADNLPProblems), [:ADNLPProblems]))
my_nlp = test.f(:sinquad);

t = R2(my_nlp)