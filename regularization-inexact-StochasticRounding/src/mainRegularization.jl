using LinearAlgebra
using Random
using Printf
using DataFrames
using Logging
using OptimizationProblems
using NLPModels
using ADNLPModels
using OptimizationProblems.ADNLPProblems
using SolverCore
using SolverBenchmark
using Plots
using Quadmath
#using BFloat16s

include("finite_precision_regularization.jl")
include("multi_precision_regularization.jl")
include("quadratic_regularization.jl")

#store number of obj. and grad evaluations with the precision level
mutable struct neval
  neval_obj::Array{Pair}
  neval_grad::Array{Pair}
end

# Parameters 
setrounding(Interval,:accurate) # rounding mode for interval evaluation
T_list = [Float32,Float64] # precision levels. /!\ use only Float32 and Float64 if IntervalArithmetic is used for obj./grad evaluation
ϵ_abs = (eps(T_list[end]))^(1/3)
ϵ_rel = (eps(T_list[end]))^(1/3)
γ_fun(n,u) = n*u # dot product error: |fl(x.y)-x.y|≤|x|.|y|γ_fun(n,u) with n dim of the problem and u machine prec
max_iter = 1000

# results storage lists
grad_error_pb=[]
R2_error_pb=[]
FR2_error_pb=[]
MR2_error_pb=[]
pb_list = []
R2_pb = []
FR2_pb = []
MR2_pb = []
R2_status = []
FR2_status = []
MR2_status = []
R2_eval = []
FR2_eval = []
MR2_eval = []
problems = setdiff(names(OptimizationProblems.ADNLPProblems), [:ADNLPProblems])

# Run R2 FR2 and MR2 over unconstrained problems
for s in problems
  nlp = eval(s)(type = Val(T_list[end]))
  if nlp.meta.ncon == 0
    try
      grad(nlp,nlp.meta.x0)
      grad(nlp,IntervalBox(nlp.meta.x0))
    catch e
      print("Warning: $(nlp.meta.name) gradient evaluation error with interval, problem skipped\n")
      push!(grad_error_pb,nlp.meta.name)
      continue
    end
    print(" =================\n $(nlp.meta.name)\n")
    push!(pb_list,nlp.meta.name)
    try
      reset!(nlp)
      stats_R2 = quadratic_regularization(
        nlp;
        ϵ_abs = ϵ_abs,
        ϵ_rel = ϵ_rel,
        maxiter = max_iter,
        verbose = false,
        log = false
      )
      push!(R2_pb,nlp.meta.name)
      push!(R2_status,stats_R2)
      push!(R2_eval,neval([Pair(nlp.counters.neval_obj,T_list[end])],[Pair(nlp.counters.neval_grad,T_list[end])]))
    catch e
     push!(R2_error_pb,nlp.meta.name)
    end
    try
      reset!(nlp)
      stats_FR2 = finite_precision_regularization(
      nlp;
      γ_n_fun = γ_fun,
      ϵ_abs = ϵ_abs,
      ϵ_rel = ϵ_rel,
      maxiter = max_iter,
      verbose = false,
      log = false
      )
      push!(FR2_pb,nlp.meta.name)
      push!(FR2_status,stats_FR2)
      push!(FR2_eval,neval([Pair(nlp.counters.neval_obj,T_list[end])],[Pair(nlp.counters.neval_grad,T_list[end])]))
    catch e
      push!(FR2_error_pb,nlp.meta.name)
    end
    try
      reset!(nlp)
      nlp_list = [eval(s)(type = Val(T)) for T ∈ T_list]
      stats_MR2 = multi_precision_regularization(
      nlp_list;
      γ_n_fun = γ_fun,
      ϵ_abs = ϵ_abs,
      ϵ_rel = ϵ_rel,
      maxiter = max_iter,
      verbose = true,
      log = false
      )
      push!(MR2_pb,nlp.meta.name)
      push!(MR2_status,stats_MR2)
      neval_obj = [Pair(nlp_list[i].counters.neval_obj,T_list[i]) for i=1:length(T_list)]
      neval_grad = [Pair(nlp_list[i].counters.neval_grad,T_list[i]) for i=1:length(T_list)]
      push!(MR2_eval,neval(neval_obj,neval_grad))
    catch e
      push!(MR2_error_pb,nlp.meta.name)
    end
  end
end

# Display status, iteration number and execution time for R2 FR2 and MR2
# Expected that FR2 and MR2 execution time are higher for FR2 and MR2 if interval evaluation is used
# Results indicated as - Inf Inf means an error occured during execution. Happens for few problems if interval evaluation used with Float32 (type unstable evaluation)

header_line = @sprintf "%15s  %13s  %5s  %7s  %13s  %5s  %7s  %13s  %5s  %7s\n" "pb name" "R2 stat" "R2 iter" "R2 time" "FR2 stat" "FR2 iter" "FR2 time" "MR2 stat" "MR2 iter" "MR2 time"
@info header_line
# open("results.txt","a") do io
#   write(io, header_line)
# end
for i in 1:length(pb_list)
  pb = pb_list[i]
  indr2 = findall(x->x==pb_list[i],R2_pb)
  if !isempty(indr2)
    r2st = R2_status[indr2[1]].status
    r2t = R2_status[indr2[1]].elapsed_time
    r2it = R2_status[indr2[1]].iter
  else
    r2st = "-"
    r2t = Inf
    r2it = Inf
  end
  indfr2 = findall(x->x==pb_list[i],FR2_pb)
  if !isempty(indfr2)
    fr2st = FR2_status[indfr2[1]].status
    fr2t = FR2_status[indfr2[1]].elapsed_time
    fr2it = FR2_status[indfr2[1]].iter
  else
    fr2st = "-"
    fr2t = Inf
    fr2it = Inf
  end
  indmr2 = findall(x->x==pb_list[i],MR2_pb)
  if !isempty(indmr2)
    mr2st = MR2_status[indmr2[1]].status
    mr2t = MR2_status[indmr2[1]].elapsed_time
    mr2it = MR2_status[indmr2[1]].iter
  else
    mr2st = "-"
    mr2t = Inf
    mr2it = Inf
  end
  infoline = @sprintf "%15s  %15s  %5d  %9.2e  %15s  %5d  %9.2e  %15s  %5d  %9.2e\n" pb_list[i] "$r2st" r2it r2t "$fr2st" fr2it fr2t "$mr2st" mr2it mr2t
  @info infoline
  # open("results.txt","a") do io
  #   write(io, infoline)
  # end
end

# compute normalized effort

R2_obj_effort = [sum([R2_eval[i].neval_obj[j][1]*4^(findall(x->x==R2_eval[i].neval_obj[j][2],T_list)[1]) for j=1:length(R2_eval[i].neval_obj)]) for i=1:length(R2_eval)]
FR2_obj_effort = [sum([FR2_eval[i].neval_obj[j][1]*4^(findall(x->x==FR2_eval[i].neval_obj[j][2],T_list)[1]) for j=1:length(FR2_eval[i].neval_obj)]) for i=1:length(FR2_eval)]
MR2_obj_effort = [sum([MR2_eval[i].neval_obj[j][1]*4^(findall(x->x==MR2_eval[i].neval_obj[j][2],T_list)[1]) for j=1:length(MR2_eval[i].neval_obj)]) for i=1:length(MR2_eval)]
R2_grad_effort = [sum([R2_eval[i].neval_grad[j][1]*4^(findall(x->x==R2_eval[i].neval_grad[j][2],T_list)[1]) for j=1:length(R2_eval[i].neval_obj)]) for i=1:length(R2_eval)]
FR2_grad_effort = [sum([FR2_eval[i].neval_grad[j][1]*4^(findall(x->x==FR2_eval[i].neval_grad[j][2],T_list)[1]) for j=1:length(FR2_eval[i].neval_obj)]) for i=1:length(FR2_eval)]
MR2_grad_effort = [sum([MR2_eval[i].neval_grad[j][1]*4^(findall(x->x==MR2_eval[i].neval_grad[j][2],T_list)[1]) for j=1:length(MR2_eval[i].neval_obj)]) for i=1:length(MR2_eval)]

# Display normalized effort
header_line = @sprintf "%15s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s\n" "pb name" "R2 stat" "R2 obj eff" "R2 grad eff" "FR2 stat" "FR2 obj eff" "FR2 grad eff"  "MR2 stat" "MR2 obj eff" "MR2 grad eff"
@info header_line
# open("results.txt","a") do io
#   write(io, header_line)
# end
for i in 1:length(pb_list)
  pb = pb_list[i]
  indr2 = findall(x->x==pb_list[i],R2_pb)
  if !isempty(indr2)
    r2st = R2_status[indr2[1]].status
    r2oe = R2_obj_effort[indr2[1]]
    r2ge = R2_grad_effort[indr2[1]]
  else
    r2st = "-"
    r2oe = 0
    r2ge = 0
  end
  indfr2 = findall(x->x==pb_list[i],FR2_pb)
  if !isempty(indfr2)
    fr2st = FR2_status[indfr2[1]].status
    fr2oe = FR2_obj_effort[indfr2[1]]
    fr2ge = FR2_grad_effort[indfr2[1]]
  else
    fr2st = "-"
    fr2oe = 0
    fr2ge = 0
  end
  indmr2 = findall(x->x==pb_list[i],MR2_pb)
  if !isempty(indmr2)
    mr2st = MR2_status[indmr2[1]].status
    mr2oe = MR2_obj_effort[indmr2[1]]
    mr2ge = MR2_grad_effort[indmr2[1]]
  else
    mr2st = "-"
    mr2oe = 0
    mr2ge = 0
  end
  maxoe = max(r2oe,fr2oe,mr2oe)
  maxge = max(r2ge,fr2ge,mr2ge)
  infoline = @sprintf "%15s  %10s  %10.3f  %10.3f  %11s  %10.3f  %10.3f  %11s  %10.3f  %10.3f\n" pb_list[i] "$r2st" r2oe/maxoe r2ge/maxge "$fr2st" fr2oe/maxoe fr2ge/maxge "$mr2st" mr2oe/maxoe mr2ge/maxge
  @info infoline
  # open("results.txt","a") do io
  #   write(io, infoline)
  # end
end