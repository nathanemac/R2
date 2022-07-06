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
using FastFloat16s

include("R2.jl")



function benchmark_problems(;type_norm::T = Val(Float32), type_sr::Tsr = Val(Float32sr)) where {T, Tsr}
    skip_list = [:ADNLPProblems, :sinquad, :NZF1, :cosine, :eg2, :fletcbv2, :fletcbv3_mod, :genhumps, :indef_mod, :noncvxu2, :noncvxun, :schmvett, :scosine, :sparsine]
    prob_rtn = (eval(problem)(type = type_norm) for problem ∈ setdiff(names(OptimizationProblems.ADNLPProblems), skip_list)) # by adding the name in the second part, it skips it 
    prob_sr =  (eval(problem)(type = type_sr) for problem ∈ setdiff(names(OptimizationProblems.ADNLPProblems), skip_list)) # by adding the name in the second part, it skips it 
    
    stats = Dict{Symbol, DataFrame}()
    stats[:Normal] = solve_problems(R2, prob_rtn,  skipif = prob -> (!unconstrained(prob) || get_nvar(prob) > 100 || get_nvar(prob) < 5))
    stats[:SR] = solve_problems(R2, prob_sr,  skipif = prob -> (!unconstrained(prob) || get_nvar(prob) > 100 || get_nvar(prob) < 5))
    
    profile = df -> df.elapsed_time
    p = performance_profile(stats, profile)
    

    return p
end


r16 = benchmark_problems(type_norm = Val(Float16), type_sr = Val(Float16sr))
r32 = benchmark_problems(type_norm = Val(Float32), type_sr = Val(Float32sr))
rb16 = benchmark_problems(type_norm = Val(BFloat16), type_sr = Val(BFloat16sr))
rf16 = benchmark_problems(type_norm = Val(FastFloat16), type_sr = Val(FastFloat16sr))



test = (eval(problem)(type = Val(Float32)) for problem ∈ setdiff(names(OptimizationProblems.ADNLPProblems), [:ADNLPProblems, :sinquad, :NZF1, :cosine, :eg2, :fletcbv2, :fletcbv3_mod, :genhumps, :indef_mod, :noncvxu2, :noncvxun, :schmvett, :scosine, :sparsine])) # by adding the name in the second part, it skips it 

for prob in test
    println(get_name(prob)) 
end