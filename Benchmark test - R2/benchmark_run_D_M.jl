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

include("R2.jl")
include("R2_D.M.jl")

function run_benchmark(;type::Val{T} = Val(Float64)) where {T}
  # load the problems:

  problems = (
    eval(problem)(type = Val(T)) for problem ∈ setdiff(names(OptimizationProblems.ADNLPProblems), [:ADNLPProblems]))


  ϵ_abs = eps(T)^1/3
  ϵ_rel = eps(T)^1/3

  # Create a dictionary for the solver: 
  solver = Dict(
          :R2_DM => 
              nlp -> quadratic_regularization(
                  nlp;
                  σ_min=nothing,
                  η_1 = 0.3,
                  η_2 = 0.7,
                  γ_1 = 1/2,
                  γ_2 = 2.0,
                  ϵ_abs = ϵ_abs,
                  ϵ_rel = ϵ_rel,
                  maxiter=1000,
                  x0=nothing,
                  verbose = false)
              ) 

  # benchmark the problems
  stats = bmark_solvers(solver, problems, skipif = prob -> (!unconstrained(prob) || get_nvar(prob) > 100 || get_nvar(prob) < 5))
  columns = [:id, :name, :nvar, :objective, :dual_feas, :neval_obj, :neval_grad, :iter, :elapsed_time, :status]
  r = stats[:R2_DM]
  res = r[!, columns]

  header = Dict(
    :nvar => "n",
    :objective => "f(x)",
    :dual_feas => "‖∇f(x)‖",
    :neval_obj => "# f",
    :neval_grad => "# ∇f",
    :neval_hprod => "# ∇²f v",
    :elapsed_time => "t",
  )
  
  for solver ∈ keys(solver)
      pretty_stats(stats[solver][!, columns], hdr_override=header)
  end
  
  return res
end

function stats_precision(df::DataFrame, precision::DataType)

  max_iter = 0
  exception = 0
  first_order = 0
  unbounded = 0
  underflow = 0
  for stat in df.status
    if stat == :max_iter
      max_iter+=1
    elseif stat == :exception
      exception +=1
    elseif stat == :first_order
      first_order +=1
    elseif stat == :unbounded
      unbounded +=1
    else 
      underflow +=1
    end
  end

  if Inf in df.elapsed_time
    filter!(e->e≠ Inf, df.elapsed_time)
  end

  return [precision, max_iter, exception, first_order, unbounded, underflow, mean(df.elapsed_time)]
end


r16 = run_benchmark(type = Val(Float16))
r32 = run_benchmark(type = Val(Float32))
r64 = run_benchmark(type = Val(Float64))
rbigfloat = run_benchmark(type = Val(BigFloat))
r128 = run_benchmark(type = Val(Float128))
rbf16 = run_benchmark(type = Val(BFloat16))
rd16 = run_benchmark(type = Val(Double16))
rd32 = run_benchmark(type = Val(Double32))
rd64 = run_benchmark(type = Val(Double64))

df = DataFrame(precision = DataType[], max_iter = Int64[], exception = Int64[], first_order = Int64[], unbounded = Int64[], underflow = Int64[], mean_time = Float64[])

push!(df, stats_precision(r16, Float16))
push!(df, stats_precision(rbf16, BFloat16))
push!(df, stats_precision(rd16, Double16))
push!(df, stats_precision(r32, Float32))
push!(df, stats_precision(rd32, Double32))
push!(df, stats_precision(r64, Float64))
push!(df, stats_precision(rd64, Double64))
push!(df, stats_precision(r128, Float128))
push!(df, stats_precision(rbigfloat, BigFloat))


CSV.write("/Users/nathanallaire/Desktop/GERAD/R2/stats_precision_DM.csv", df)