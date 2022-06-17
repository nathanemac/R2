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

include("R2.jl")
include("R2_D.M.jl")

function run_benchmark(;type::Val{T} = Val(Float64)) where {T}

  # load the problems
  problems = (eval(problem)(type = Val(T)) for problem ∈ setdiff(names(OptimizationProblems.ADNLPProblems), [:ADNLPProblems]))



  # Create a dictionary for the solver: 
  solvers = Dict(
            :R2 => 
              nlp -> R2(
                  nlp;
                  maxiterations=250,
                  η1 = 1e-4,
                  η2 = 0.75,
                  γ1 = 0.33,
                  γ2 = 2.0,
                  verbose = false),
            :R2_DM =>
              nlp -> quadratic_regularization(
                nlp;
                σ_min=nothing,
                η_1 = 1e-4,
                η_2 = 0.75,
                γ_1 = 0.33,
                γ_2 = 2.0,
                x0=nothing,
                maxiter = 250,
                verbose = false,
                log = false),
              )


  # benchmark the problems
  stats = bmark_solvers(solvers, problems, skipif = prob -> (!unconstrained(prob) || get_nvar(prob) > 100 || get_nvar(prob) < 5))
  columns = [:id, :name, :nvar, :objective, :dual_feas, :neval_obj, :neval_grad, :iter, :elapsed_time, :status]

  header = Dict(
    :nvar => "n",
    :objective => "f(x)",
    :dual_feas => "‖∇f(x)‖",
    :neval_obj => "# f",
    :neval_grad => "# ∇f",
    :neval_hprod => "# ∇²f v",
    :elapsed_time => "t",
  )

  for solver ∈ keys(solvers)
    pretty_stats(stats[solver][!, columns], hdr_override=header)
  end
  r2 = stats[:R2]
  r2dm = stats[:R2_DM]
  resr2 = r2[!, columns]
  resr2dm  = r2dm[!, columns]

  profile = df -> df.elapsed_time

  p = performance_profile(stats, profile)


  return resr2, resr2dm, p
end

function stats_precision(df::DataFrame, precision::DataType)

  max_iter = 0
  exception = 0
  first_order = 0
  unbounded = 0
  small_step = 0
  overflow = 0
  for stat in df.status
    if stat == :max_iter
      max_iter+=1
    elseif stat == :exception
      exception +=1
    elseif stat == :first_order
      first_order +=1
    elseif stat == :unbounded
      unbounded +=1
    elseif stat == small_step
      small_step +=1
    else 
      overflow+=1
    end
  end

  if Inf in df.elapsed_time
    filter!(e->e≠ Inf, df.elapsed_time)
  end

  return [precision, max_iter, exception, first_order, unbounded, small_step, mean(df.elapsed_time)]
end



################# STATS ON THE SOLVERS #########################

r16 = run_benchmark(type = Val(Float16))
r32 = run_benchmark(type = Val(Float32))
r64 = run_benchmark(type = Val(Float64))
rbigfloat = run_benchmark(type = Val(BigFloat))
r128 = run_benchmark(type = Val(Float128))
rbf16 = run_benchmark(type = Val(BFloat16))
rd16 = run_benchmark(type = Val(Double16))
rd32 = run_benchmark(type = Val(Double32))
rd64 = run_benchmark(type = Val(Double64))

R_performance = [r16[3], r32[3], r64[3], rbigfloat[3], r128[3], rbf16[3], rd16[3], rd32[3], rd64[3]]


df = DataFrame(precision = DataType[], max_iter = Int64[], exception = Int64[], first_order = Int64[], unbounded = Int64[], small_step = Int64[], mean_time = Float64[])
push!(df, stats_precision(r16[1], Float16))
push!(df, stats_precision(rbf16[1], BFloat16))
push!(df, stats_precision(rd16[1], Double16))
push!(df, stats_precision(r32[1], Float32))
push!(df, stats_precision(rd32[1], Double32))
push!(df, stats_precision(r64[1], Float64))
push!(df, stats_precision(rd64[1], Double64))
push!(df, stats_precision(r128[1], Float128))
push!(df, stats_precision(rbigfloat[1], BigFloat))



df2 = DataFrame(precision = DataType[], max_iter = Int64[], exception = Int64[], first_order = Int64[], unbounded = Int64[], small_step = Int64[], mean_time = Float64[])
push!(df2, stats_precision(r16[2], Float16))
push!(df2, stats_precision(rbf16[2], BFloat16))
push!(df2, stats_precision(rd16[2], Double16))
push!(df2, stats_precision(r32[2], Float32))
push!(df2, stats_precision(rd32[2], Double32))
push!(df2, stats_precision(r64[2], Float64))
push!(df2, stats_precision(rd64[2], Double64))
push!(df2, stats_precision(r128[2], Float128))
push!(df2, stats_precision(rbigfloat[2], BigFloat))


CSV.write("/Users/nathanallaire/Desktop/GERAD/R2/Benchmark test - R2/stats_250_R2.csv", df)
CSV.write("/Users/nathanallaire/Desktop/GERAD/R2/Benchmark test - R2/stats_iter250_R2_DM.csv", df2)



################################# TIME PROFILE FOR QUADRUPLE PRECISION ######################################
#TODO do this 