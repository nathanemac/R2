
"""
Author: D. Monnet and F. Rahbarnia

Finite precision regularization algorithm.
"""

using IntervalArithmetic
using ForwardDiff

# convertion of IntervalBox to vector of interval, needed for gradient evaluation
Vector(B::IntervalBox) = [X for X in B]


function ForwardDiff.gradient(f, X :: IntervalBox)
  v = Vector(X)
  ib = ForwardDiff.gradient(f, v)
  return ib
end

#overload NLPModels obj and grad evaluation with IntervalBox

function NLPModels.obj(nlp::ADNLPModel,X :: IntervalBox)
  v = Vector(X)
  return obj(nlp,v)
end

function NLPModels.grad(nlp::ADNLPModel,X::IntervalBox)
  nlp.counters.neval_grad+=1
  return ForwardDiff.gradient(nlp.f,X)
end

function multi_precision_regularization(
  nlp_list;
  γ_n_fun = nothing, # the \gmma_n function
  ωg_fun_list = nothing,
  ωf_fun_list = nothing,
  x0 = nothing,
  σ_min = 2^(-5),
  η_0 = 0.1,
  η_1 = 0.3,
  η_2 = 0.4,
  γ_1 = 2^(-1),
  γ_2 = 2.0,
  κ_μ = 0.2,
  ϵ_abs,
  ϵ_rel,
  maxiter = 1000,
  verbose::Bool = false,
  log::Bool = false
)
  # print problem name
  verbose && @info "" nlp_list[1].meta.name
  if log
    open("output.txt","a") do io
      write(io, nlp_list[1].meta.name)
    end
  end

  # get precisions nlp_list
  type_list = get_type_list(nlp_list)

  # declare precisions and initialize default value
  πx = 1
  πg = 1
  πf = 1
  πfm = 1
  πc = 1
  πmax = length(nlp_list)


  # Define arguments not provided
  if x0 === nothing
    x0 = copy(nlp_list[1].meta.x0)
  elseif type(x0[1]) ∉ type_list[1]
    @warn "Initial point provided does not match any of the provided precision levels, converted to lowest precision level: $(type_list[1])"
    x0 = convert.(type_list[1],x0)
  end
  
  if σ_min === nothing
    σ_min = eps(type_list[1])
  else 
    σ_min = convert(type_list[1],σ_min)
  end

  if γ_n_fun === nothing
    @info "No γₙ model provided, γₙ = nu used by default"
    γ_n_fun = function(n,u) return n*u end
  end

  # Interval evaluation is used by default if no error model is provided
  # abort if interval evaluation causes error or is not type stable (returned interval's type is different than input)
  ωf_int = false
  if ωf_fun_list === nothing
    @info "No objective function error model provided, interval evaluation will be used by default"
    try 
      for nlp in nlp_list
        X0 = IntervalBox(nlp.meta.x0)
        intype = typeof(nlp.meta.x0[1])
        outtype = typeof(obj(nlp,X0)[1].lo)
        if intype != outtype
          @warn("Interval evaluation of objective function not type stable ($intype -> $outtype)")
          error()
        end
      end
    catch e
      error("Objective function cannot be evaluated with interval, error model must be provided")
    end
    ωf_int = true
  end
  
  ωg_int = false
  if ωg_fun_list === nothing
    @info "No gradient error model provided, interval evaluation will be used by default"
    try 
      for nlp in nlp_list
        X0 = IntervalBox(nlp.meta.x0)
        intype = typeof(nlp.meta.x0[1])
        outtype = typeof(grad(nlp,X0)[1].lo)
        if intype != outtype
          @warn("Interval evaluation of gradient not type stable ($intype -> $outtype)")
          error()
        end
      end
    catch e
      error("Gradient cannot be evaluated with interval, error model must be provided")
    end
    ωg_int = true
  end

  # reset obj and grad counters
  for i in 1:length(nlp_list)
    reset!(nlp_list[i])
  end
  
  #set dot product error related quantities
  n = length(x0)
  u_list = eps.(type_list) # can take eps(typeof(x))/2 if rounding to the nearest
  γ_n2_list = γ_n_fun.(n+2,u_list)
  α_n_list = 1 ./(1 .- γ_n2_list)
  γ_norm_list = γ_n_fun.(ceil(n/2)+1,u_list)

  #convert gamma and sigma to the lowest precision level
  γ_1 =  convert(type_list[1],γ_1)
  γ_2 =  convert(type_list[1],γ_2)
  σ_min =  convert(type_list[1],σ_min)

  # Check the conditions and send error 
  0 < γ_1 || error("γ_1 must be > 0")
  0 < γ_2 || error("γ_2 must be > 0")
  γ_1 ≤ 1 || error("γ_1 must be <= 1")
  γ_1 < 1 < γ_2 || error("0 < γ_1 < 1 < γ_2 ")
  0 < η_1 ≤ η_2 < 1 || error("must be 0 <η_1 ≤ η_2 < 1")
  η_0 ≤ (0.5) * η_1 || error("must follow η_0 ≤(0.5) η_1")
  2 * η_0 + κ_μ ≤ 1 - η_2 || error("must follow 2*η_0 + κ_μ  ≤ 1-η_2 ")
  
  # initialize variables
  iter = 0
  xk = copy(x0)
  μk = 0
  ρk = 0

  # flags for lost of convergence property
  μk_conv = true
  ωf_conv = true

  # flag for xk and sk convertion to πc if step is too small
  convert_to_pi_c = false

  # Evaluate the objective fonction and its derivative at the initial point
  ck = copy(xk)
  fk,ωfk,ωf_conv,πfm = evaluate_obj_n_error(nlp_list,xk,πfm,-1,ωf_int,ωf_fun_list)
  gk,ωgk = evaluate_grad_n_error(nlp_list,ck,πg,ωg_int,ωg_fun_list)
  gk_norm = norm(gk)
  σ_0 = 2^round(log2(gk_norm+1)) # ensures ||s_0|| ≈ 1 with sigma_0 = 2^n with n an interger, i.e. sigma_0 is exactly representable in n bits 
  σk = convert(type_list[1],σ_0) # make sure σk is representable in lowest precision level

  # print algo param
  verbose && @info "" nlp_list[1].meta.nvar σ_0 σ_min η_0 η_1 η_2 γ_1 γ_2

  # print @info header
  if verbose
    infoline = @sprintf "%5s  %9s  %9s  %9s  %9s  %7s  %7s  %7s  %7s  %9s  %7s  %2s  %2s  %2s  %2s\n" "iter" "fk" "ωfxk" "f(ck)" "ωfck" "‖gk‖" "σk" "μk" "ρk" "||sk||" "ϕk" "πx" "πg" "πc" "πf"
    @info infoline
    if log
      open("output.txt","a") do io
        write(io, infoline)
      end
    end
  end

  # Define the stopping criteria
  ϵ = ϵ_abs + ϵ_rel * gk_norm
  optimal = gk_norm ≤ (1-γ_norm_list[πg])*ϵ/(1+ωgk)
  tired = iter > maxiter

  # start timer
  status=nothing
  start_time = time()
  elapsed_time = 0.0

  while !(optimal | tired)
    # Step and candidate computations
    sk = - gk./σk 
    if norm(sk) <eps(type_list[πx]) # this case should somewhat be avoided with the condition μk <= κ_μ but might still happen
      prec = findall(x->x<(norm(sk)),eps.(type_list))
      if isempty(prec) # no precision available such that step size is big enough compared to machine prec
        @warn "Algo stops because the step size is lower than machine precision of highest precision level"
        status = :small_step
        break
      else # convert preform xk + sk with high enough precision
        p = prec[1] # choose lowest precision for which step size is accaptable
        πc = max(πc,p) # should always be p
        convert_to_pi_c = true
      end
    end
    if convert_to_pi_c
      πx_for_c = max(πc,πx) # should always be πc
      πs = findall(x->x==typeof(sk[1]),type_list)[1]
      πs_for_c = max(πc,πs)
      ck = convert.(type_list[πc],convert.(type_list[πx_for_c],xk) .+ convert.(type_list[πs_for_c],sk))
    else 
      ck = convert.(type_list[πc],xk .+ sk)
    end


    # Check stopping criterion on μ for lost of convergence 
    ϕ_hat_k = norm(xk)/norm(sk)
    ϕk = ϕ_hat_k * (1+γ_norm_list[πx])/(1-γ_norm_list[πg])
    if convert_to_pi_c
      u = 2*eps(type_list[πc]) + eps(type_list[πc])^2
    else
      u = eps(type_list[πg]) + eps(type_list[πc]) + eps(type_list[πg])*eps(type_list[πc])
    end
    μk = (α_n_list[πg]*ωgk*(u+ϕk*u+1) + u*(ϕk*α_n_list[πg] + α_n_list[πg] + 1) +γ_n2_list[πg]/(1+γ_n2_list[πg]))/(1+u)
    if μk > κ_μ
      if πc == πmax && πg == πmax
        μk_conv = false
        @warn "Convergence not ensured: evaluation error of g(xk) too big: μ = $μk > $κ_μ = κ_μ"
      elseif πc<πg || πg == πmax
        πc = min(πmax,πc+1)
        continue
      else 
        πg_new = min(πg+1,πmax)
        if πg_new!=πg # restart loop if maximum precision has not already been reached for gradient evaluation
          πg = πg_new
          gk,ωgk = evaluate_grad_n_error(nlp_list,xk,πg,ωg_int,ωg_fun_list)
        end
        continue
      end
    elseif !μk_conv
      μk_conv = true
    end

    # Compute approximated taylor series decrease
    ΔTk= -gk' * sk
    
    # Evaluate objective function at xk if necessary
    if ωfk>η_0*ΔTk
      πfm_new = min(πmax, πfm + 1) # increase precision evaluation at xk
      if πfm_new != πfm # re-evaluate only if max precision has not already been reached
        πfm = πfm_new
        fk,ωfk,ωf_conv,πfm = evaluate_obj_n_error(nlp_list,xk,πfm,η_0*ΔTk,ωf_int,ωf_fun_list)
      end
    end

    # Evaluate objective function at ck
    πf = πc
    ftrial,ωftrial,ωf_conv,πf = evaluate_obj_n_error(nlp_list,ck,πf,η_0*ΔTk,ωf_int,ωf_fun_list)
    

    # Compute ρₖ
    if(fk==ftrial)
      @warn "fk==ftrial"
    end
    ρk = (fk - ftrial)/ΔTk 

    # σk update
    if ρk ≥ η_2 
      σk = max(σ_min,  σk*γ_1)
    elseif ρk < η_1
      σk = σk * γ_2
    end

    if(σk==Inf)
      error("σk = Inf")
    end

    # print info
    if verbose || log
      infoline = @sprintf "%5d  %9.2e  %9.2e  %9.2e  %9.2e  %7.1e  %7.1e  %7.1e  %7.1e  %9.2e  %7.1e  %2d  %2d  %2d  %2d  \n" iter fk ωfk ftrial ωftrial gk_norm σk μk ρk norm(sk) ϕk πx πg πc πf
      if log
        open("output.txt","a") do io
          write(io, infoline)
        end
      end
      @info infoline
    end

    # Acceptance of the trial point (Accepting the step)
    if ρk ≥ η_1
      πg = πc
      gk,ωgk = evaluate_grad_n_error(nlp_list,ck,πg,ωg_int,ωg_fun_list)
      gk_norm = norm(gk)
      xk = ck
      fk = ftrial
      ωfk = ωftrial
      πfm = πf
      πx = πc
      πc = max(1,πf-1) # arbitrary strategy
    end

    iter += 1
    optimal = gk_norm ≤1/(1+γ_norm_list[πg])*ϵ/(1+ωgk)
    tired = iter > maxiter
  end
  elapsed_time = time()-start_time
  if optimal
    status = :first_order
  elseif tired
    status = :max_iter
  elseif status === nothing
    status = :exception
  end

  # types must match here to avoid GenericExecutionStats error
  mprec = max(πx,πf,πg)
  xk = convert.(type_list[mprec],xk)
  fk = convert(type_list[mprec],fk)
  gk_norm = convert(type_list[mprec],gk_norm)
  
  return GenericExecutionStats(
    status,
    nlp_list[mprec],
    solution = xk,
    objective = fk,
    dual_feas = gk_norm,
    elapsed_time = elapsed_time,
    iter = iter,
  )
end

# Evaluate objective function and error and decrease prec until error small enough, return error message if error is too big
function evaluate_obj_n_error(nlp_list,xk,πf,bound,ωf_int,ωf_fun_list)
  fk = nothing
  ωfk = nothing
  need_eval = true
  ωf_conv = true
  πmax = length(nlp_list)
  while need_eval
    if ωf_int
      # cast xk as interval vector for interval evaluation
      type = typeof(nlp_list[πf].meta.x0[1])
      xk = convert.(type,xk)
      Xk = IntervalBox(xk)
      Fk = obj(nlp_list[πf],Xk)
      fk = mid(Fk)
      ωfk = radius(Fk)
    else
      ωfk =ωf_fun_list[πf](xk)
    end
    if ωfk <= bound || bound<0 # second condition used to evaluate objective function regardless the bound
      if !ωf_int
        fk = obj(nlp_list[πf],xk)
      end
      return fk,ωfk,ωf_conv,πf
    else
      if πf == πmax
        ωf_conv = false
        @warn "Convergence not ensured: evaluation error of f too big: ωf = $(ωfk) > $bound = η_0*ΔTk "
        return fk, ωfk,ωf_conv,πf
      else
        πf = πf+1
        need_eval = ωfk > bound
      end
    end
  end
end

# Evaluate gradient and error, deal with the case gk = 0
function evaluate_grad_n_error(nlp_list,ck,πg,ωg_int,ωg_fun_list)
  gk = nothing
  ωgk = nothing
  if ωg_int
    type = typeof(nlp_list[πg].meta.x0[1])
    ck = convert.(type,ck)
    Ck = IntervalBox(ck)
    Gk = grad(nlp_list[πg],Ck)
    gk = mid.(Gk)
    gk_norm = norm(gk)
    # cover the case gk = 0
    if gk_norm == 0
      if sum(radius.(Gk)) == 0 # Gk = [0,0]ⁿ
        ωgk = 0
      else # take upper bounds of Gk by default for gk
        gk = [Gk[i].hi for i=1:length(Gk)]
        gk_norm = norm(gk)
        ωgk = norm(Gk)/gk_norm
      end
    else
      ωgk = norm(radius.(Gk))/gk_norm
    end
  else
    ωgk =ωg_fun_list[πg](ck)
    gk = grad(nlp_list[πg],ck)
  end
  return gk,ωgk
end

function get_type_list(nlp_list)
  type_list = []
  for nlp in nlp_list
    push!(type_list,typeof(nlp.meta.x0[1]))
  end
  if sort(eps.(type_list)) != reverse(eps.(type_list))
    error("Input argument: nlp_list should be ordered by increasing precision levels (Float32,Float64,...)")
  end
  return type_list
end
