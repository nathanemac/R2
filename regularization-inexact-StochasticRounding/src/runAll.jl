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

function Run_experiment(nlp, nlp_sr, T1, T2, maxiter = 50, verbose::Bool = false, maxRun = 20)
  ϵ_abs = (eps(T1))^(1 / 3)
  ϵ_rel = (eps(T1))^(1 / 3)

  stats = adaptative_regularization(
    nlp;
    σ_min = 1.0,
    η_1 = 1e-4,
    η_2 = 0.75,
    γ_1 = 0.33,
    γ_2 = 1,
    γ_3 = 5,
    ϵ_abs = ϵ_abs, #TODO dependes on the accracy 
    ϵ_rel = ϵ_rel,
    maxiter = maxiter,
    verbose = verbose,
  )

  ϵ_abs = (eps(T2))^(1 / 3) #TODO make it smaller
  ϵ_rel = (eps(T2))^(1 / 3)

  x = Float64[]
  y = Float64[]
  for i = 1:maxRun
    push!(x, i)
    out_put = adaptative_regularization(
      nlp_sr;
      σ_min = 1.0,
      η_1 = 1e-4,
      η_2 = 0.75,
      γ_1 = 0.33,
      γ_2 = 1,
      γ_3 = 5,
      ϵ_abs = ϵ_abs, #TODO dependes on the accracy 
      ϵ_rel = ϵ_rel,
      maxiter = maxiter,
      verbose = verbose,
    )
    rel_err = norm(Float32.(out_put.solution) - stats.solution) / norm(stats.solution)
    push!(y, rel_err)
    #   println(x, y)
  end

  # Plot 
  nlp_title = string("Relative Error of the Float32sr vs Float32 for problem: ", nlp.meta.name)
  plot(x, y, title = nlp_title, titlefontsize = 10, label = "Err")
  savefig(string("plots/", nlp.meta.name, "_SR_Err", ".png"))
end

T = Float32
T2 = Float32sr

problems = ( #TODO Make sure the problem reach the stopping condition 
  (eval(problem)(type = Val(T)), eval(problem)(type = Val(T2))) for
  problem ∈ setdiff(names(OptimizationProblems.ADNLPProblems), [:ADNLPProblems])
)
# TODO save logger for functions 
# Logger to the file --Julia logger
for prob in problems
  if (!unconstrained(prob[1]) || get_nvar(prob[1]) > 100 || get_nvar(prob[1]) < 5)
    continue
  end
  Run_experiment(prob[1], prob[2], T, T2)
end
