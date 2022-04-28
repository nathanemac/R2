using LinearAlgebra
using Random
using Printf
using DataFrames
using Logging
using StochasticRounding
using OptimizationProblems
using NLPModels
using ADNLPModels
using OptimizationProblems.ADNLPProblems
using SolverCore
using SolverBenchmark
using Plots

include("tools/inexact_evaluation.jl")
include("adaptative_regularization.jl")
include("multi_precision_regularization.jl")
include("tools/normest.jl")



#TODO 
#T = Float16
# T = Float32sr
T = Float32
# T = Float64

ϵ_abs = (eps(T))^(1 / 3)
ϵ_rel = (eps(T))^(1 / 3)
print(ϵ_abs)

nlp = rosenbrock(type = Val(T),n=2)

γ_fun(n,u) = n*u
stats = multi_precision_regulazation( #TODO change this
  nlp;
  γ_n_fun =γ_fun,
  ωf_fun = 0,
  ωg_fun = 0,
  ϵ_abs = ϵ_abs, #TODO dependes on the accracy 
  ϵ_rel = ϵ_rel,
  maxiter = 10000,
  verbose = true,
)



# here we compute the Float32sr 20 times 
# T = Float32sr
T = Float16sr

ϵ_abs = (eps(T))^(1 / 3) #TODO make it smaller
ϵ_rel = (eps(T))^(1 / 3)
print(ϵ_abs)

nlp_sr = arglina(type = Val(T))

x = Float64[]
y = Float64[]
for i = 1:5
  push!(x, i)
  out_put = adaptative_regularization(
    nlp_sr;
    σ_min = 2^(-5),
    η_1 = 0.1,
    η_2 = 0.4,
    γ_1 = 2^-1,
    γ_2 = 2,
    ϵ_abs = ϵ_abs, #TODO dependes on the accracy 
    ϵ_rel = ϵ_rel,
    maxiter = 500,
    verbose = true,
  )
  rel_err = norm(Float32.(out_put.solution) - stats.solution) / norm(stats.solution)
  push!(y, rel_err)
  println(x, y)
end

# Plot 
nlp_title = string("Relative Error of the Float32sr vs Float32 for problem: ", nlp.meta.name)
plot(x, y, title = nlp_title, titlefontsize = 10, label = "Err")
savefig(string("plots/", nlp.meta.name, "_SR_Err", ".png"))
