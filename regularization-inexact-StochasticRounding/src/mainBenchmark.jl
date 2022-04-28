using LinearAlgebra
using Random
using Printf
using DataFrames
using Logging
using StochasticRounding
using OptimizationProblems
using NLPModels
using NLPModelsJuMP
using ADNLPModels
using OptimizationProblems.ADNLPProblems
using NLPModels
using SolverCore
using SolverBenchmark
using DelimitedFiles
using Arpack

include("tools/inexact_evaluation.jl")
include("tools/normest.jl")
include("adaptative_regularization.jl")

"""
This function is used to get the list of solvers and write it into a file
"""
function solved_problems(
  stats::Dict{Symbol, DataFrame},
  filename::AbstractString;
  force::Bool = false,
  # solver,
)
  df = stats[:adaptative_regularization] #TODO should I do a deep copy
  names = df[df[!, :status] .== :first_order, :][!, :name]
  isfile(filename) && !force && error("$filename already exists; use `force=true` to overwrite")
  writedlm(filename, names)
end
"""
This function takes a file name, if it doesn't exists return an empty list
"""
function read_names(filename::AbstractString)
  name_list = String[]
  if (isfile(filename))
    open(filename, "r") do f
      for ln in eachline(f)
        push!(name_list, ln)
      end
    end
  else
    return String[]
  end
  return name_list
end
# TODO figure a way to change the type ?
#TODO 
T = Float32
# T = Float32sr
# T = Float64

ϵ_abs = (eps(T))^(1 / 3)
ϵ_rel = (eps(T))^(1 / 3)

problems = (
  eval(problem)(type = Val(T)) for
  problem ∈ setdiff(names(OptimizationProblems.ADNLPProblems), [:ADNLPProblems])
)
# dict of solvers , for now we only have one 
solvers = Dict(
  :adaptative_regularization =>
    nlp -> adaptative_regularization(
      nlp;
      # nlp_helper= nlp 32bit
      σ_min = 1.0,
      η_1 = 1e-4,
      η_2 = 0.75,
      γ_1 = 0.33,
      γ_2 = 1,
      γ_3 = 5,
      ϵ_abs = ϵ_abs, #1e-6, #TODO relax to bigger
      ϵ_rel = ϵ_rel, #1e-6,
      maxiter = 500, #TODO smaller ~ 500 , how many do we solve
      verbose = false,
    ),
)
namesList = read_names("solved.txt")
if isempty(namesList)
  stats = bmark_solvers(
    solvers,
    problems,
    skipif = prob -> (!unconstrained(prob) || get_nvar(prob) > 100 || get_nvar(prob) < 5), #TODO check this 
  )

else #if the list of names isn't empty that means we ran it once, and filtered the ones that reached the solution 
  stats = bmark_solvers(
    solvers,
    problems,
    skipif = prob -> (
      !unconstrained(prob) ||
      !(get_name(prob) ∈ namesList) ||
      get_nvar(prob) > 100 ||
      get_nvar(prob) < 5
    ),
  )
end

pretty_stats(stats[:adaptative_regularization])

statuses, avgs = quick_summary(stats)
for solver ∈ keys(stats)
  @info "statistics for" solver statuses[solver] avgs[solver]
end

solved_problems(stats, "solved.txt", force = true)
# TODO save stats , save them so we can compare the results 
