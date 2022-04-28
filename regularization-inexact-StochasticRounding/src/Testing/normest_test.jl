using Test
using Arpack
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
# Import the other files    
include("../tools/normest.jl")
# TODO fix The Problem is: genhumps_autodiff
function comp_Arpack(ϵ)
  # Get The problems
  problems = (
    eval(problem)() for
    problem ∈ setdiff(names(OptimizationProblems.ADNLPProblems), [:ADNLPProblems])
  )
  for prob in problems
    if (!unconstrained(prob) || get_nvar(prob) > 100 || get_nvar(prob) < 5)
      continue
    end
    println(string("The Problem is: ", prob.meta.name, " \t"))

    buffer = hess_op(prob, copy(prob.meta.x0))
    σ_0_normest, _ = normest(buffer, 1.0e-4, 10000)

    λ, ϕ = eigs(buffer, which = :LM) # norm of ∇2f(x_0) getting the largest Eigenvalue 
    σ_0_Arpack = abs(Float64(λ[1]))

    println("---------------------------------------\n")
    println("Testing the σ_0 calculated using normest vs Arpack ")
    println(string("The Problem is: ", prob.meta.name, " \t"))
    println(string("σ_0_normest: ", σ_0_normest, " \t"))
    println(string("σ_0_Arpack: ", σ_0_Arpack, " \t"))

    # println(@test abs(σ_0_normest - σ_0_Arpack) <= ϵ)
    if σ_0_Arpack == 0
      dev = 1
    else
      dev = abs(σ_0_Arpack)
    end
    println(@test abs(σ_0_normest - σ_0_Arpack) / dev <= ϵ)
  end
end

function simple_matrix_test_helper(S, ϵ)
  val = opnorm(S, 2) #TODO fails for Float32sr
  val_normest, _ = normest(S, 1.0e-4, 10000)

  if val == 0
    dev = 1
  else
    dev = abs(val)
  end

  # println(string("Testing matrix norm using type ", eltype(S), " -- and size ", size(S)))
  # println(string("normest ", val_normest, " -- and opnorm ", val))

  @test abs(val_normest - val) / dev <= ϵ
end

function simple_matrix_test(ϵ)
  # S = randn(4, 4)
  S= reshape(collect(1:16), (4,4))
  simple_matrix_test_helper(S, ϵ)

  S= reshape(collect(1:160), (40,4))

  simple_matrix_test_helper(S, ϵ)

  S= reshape(collect(1:400), (20,20))
  #TODO matrix 1 to -1 of 100 then 10X10 or 20X5

  simple_matrix_test_helper(S, ϵ)
  # TODO fails for all ones 
  S = ones(Float32, 4, 4)

  S = randn(Float32, 4, 4) #TODO choose fix matrix random is no good 
  simple_matrix_test_helper(S, ϵ)

  # S = Float32sr.(randn(Float32, 4, 4))
  # simple_matrix_test_helper(S,ϵ)
end

function RunAllTest()
  thershold = 0.01
  simple_matrix_test(thershold)
  # comp_Arpack(thershold)
  # TODO check the type
end

#running the tests
RunAllTest()
