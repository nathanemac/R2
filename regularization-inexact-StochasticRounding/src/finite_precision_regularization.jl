
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


function finite_precision_regularization(
  nlp;
  γ_n_fun = nothing, # the \gmma_n function
  ωg_fun = nothing,
  ωf_fun = nothing,
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

  # print algo name
  verbose && @info "" nlp.meta.name
  if log
    open("output.txt","a") do io
      write(io, nlp.meta.name * "\n")
    end
  end

  # Define arguments not provided
  type = typeof(nlp.meta.x0[1])
  if x0 === nothing
    x0 = copy(nlp.meta.x0)
  elseif type(x0[1]) != type
    @warn "Initial point provided converted to $type"
    x0 = convert.(type,x0)
  end
  
  if σ_min === nothing
    σ_min = eps(type)
  else 
    σ_min = convert(type,σ_min)
  end

  if γ_n_fun === nothing
    @info "No γₙ model provided, γₙ = nu used by default"
    γ_n_fun = function(n,u) return n*u end
  end

  # Interval evaluation is used by default if no error model is provided
  # abort if interval evaluation causes error or is not type stable (returned interval's type is different than input)
  ωf_int = false
  if ωf_fun === nothing
    @info "No objective function error model provided, interval evaluation will be used by default"
    try 
      X0 = IntervalBox(x0)
      if typeof(obj(nlp,X0).lo) != type
        @warn("Interval evaluation of objective function not type stable")
        error()
      end
    catch e
      error("Objective function cannot be evaluated with interval, error model must be provided")
    end
    ωf_int = true
  end

  ωg_int = false
  if ωg_fun === nothing
    @info "No gradient error model provided, interval evaluation will be used by default"
    try 
      X0 = IntervalBox(x0)
      if typeof(grad(nlp,X0)[1].lo) != type
        @warn("Interval evaluation of gradient not type stable")
        error()
      end
    catch e
      error("Gradient cannot be evaluated with interval, error model must be provided")
    end
    ωg_int = true
  end

  # reset obj and grad counters
  reset!(nlp)
  
  #set dot product error related quantities
  n = length(x0)
  u = eps(type) # can take eps(typeof(x))/2 if rounding to the nearest
  γ_n2 = γ_n_fun(n+2,u)
  α_n = 1/(1-γ_n2)
  γ_norm = γ_n_fun(ceil(n/2)+1,u)

  #convert gamma and sigma to the correct type
  γ_1 =  convert(type,γ_1)
  γ_2 =  convert(type,γ_2)
  σ_min =  convert(type,σ_min)

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
  xk = x0
  μk = 0
  ρk = 0

  # flags for lost of convergence property
  μk_conv = true
  ωf_conv = true

  # Evaluate the objective fonction and its derivative at the initial point
  ck = copy(xk)
  fk,ωfk,ωf_conv = evaluate_obj_n_error(nlp,xk,Inf,ωf_int,ωf_fun)
  gk,ωgk = evaluate_grad_n_error(nlp,ck,ωg_int,ωg_fun)
  gk_norm = norm(gk)
  σ_0 = 2^round(log2(gk_norm+1)) # ensures ||s_0|| ≈ 1 with sigma_0 = 2^n with n an interger, i.e. sigma_0 is exactly representable in n bits 
  σk = convert(type,σ_0)

  # print algo param
  verbose && @info "" nlp.meta.name nlp.meta.nvar σ_0 σ_min η_0 η_1 η_2 γ_1 γ_2

  # print @info header
  if verbose
    infoline = @sprintf "%5s  %9s %9s  %9s  %9s  %7s  %7s  %7s  %7s  %9s  %7s" "iter" "fk" "ωfxk" "f(ck)" "ωfck" "‖gk‖" "σk" "μk" "ρk" "||sk||" "ϕk\n"
    @info infoline
    if log
      open("output.txt","a") do io
        write(io, infoline)
      end
    end
  end

  # Define the stopping criteria
  ϵ = ϵ_abs + ϵ_rel * gk_norm
  optimal = gk_norm ≤ (1-γ_norm)*ϵ/(1+ωgk)
  tired = iter > maxiter

  # start timer
  status=nothing
  start_time = time()
  elapsed_time = 0.0

  while !(optimal | tired)    
    # Step and candidate computations
    sk = - gk./σk 
    if norm(sk) <eps(type)
      @warn "Algo stops because the step is lower than machine precision"
      status = :small_step
      break
    end
    ck = xk .+ sk

    # Check stopping criterion on μ for lost of convergence 
    ϕ_hat_k = norm(xk)/norm(sk)
    ϕk = ϕ_hat_k * (1+γ_norm)/(1-γ_norm)
    μk = (α_n*ωgk*(u+ϕk*u+1) + u*(ϕk*α_n + α_n + 1) +γ_n2/(1+γ_n2))/(1+u)
    if ( μk > κ_μ)
      μk_conv = false
      @warn "Convergence not ensured: evaluation error of g(xk) too big: μ = $μk > $κ_μ = κ_μ"
      #error("Convergence not ensured: evaluation error of g(xk) too big: μ = $μk > $κ_μ = κ_μ")
    elseif !μk_conv
      μk_conv = true
    end

    # Compute approximated taylor series decrease
    ΔTk= -gk' * sk

    # Evaluate objective function at ck
    ftrial,ωftrial,ωf_conv = evaluate_obj_n_error(nlp,ck,η_0*ΔTk,ωf_int,ωf_fun)
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

    # Acceptance of the trial point (Accepting the step)
    if ρk ≥ η_1
      gk,ωgk = evaluate_grad_n_error(nlp,ck,ωg_int,ωg_fun)
      gk_norm = norm(gk)
      xk = ck
      fk = ftrial
      ωfk = ωftrial
    end

    iter += 1
    optimal = gk_norm ≤1/(1+γ_norm)*ϵ/(1+ωgk)
    tired = iter > maxiter

    # print info
    if verbose
      infoline = @sprintf "%5d  %9.2e %9.2e %9.2e %9.2e  %7.1e  %7.1e  %7.1e  %7.1e  %9.2e  %7.1e \n" iter fk ωfk ftrial ωftrial gk_norm σk μk ρk norm(sk) ϕk
      if log
        open("output.txt","a") do io
          write(io, infoline)
        end
      end
      @info infoline
    end
  end

  elapsed_time += time() - start_time
  if optimal
    status = :first_order
  elseif tired
    status = :max_iter
  elseif status === nothing
    status = :exception
  end

  return GenericExecutionStats(
    status,
    nlp,
    solution = xk,
    objective = fk,
    dual_feas = gk_norm,
    elapsed_time = elapsed_time,
    iter = iter,
  )
end

# Evaluate objective function and error, return error message if error is too big
function evaluate_obj_n_error(nlp,xk,bound,ωf_int,ωf_fun)
  fk = nothing
  ωfk = nothing
  ωf_conv = true
  if ωf_int
    # cast xk as interval vector for interval evaluation
    Xk = IntervalBox(xk)
    Fk = obj(nlp,Xk)
    fk = mid(Fk)
    ωfk = radius(Fk)
  else
    ωfk =ωf_fun(xk)
  end
  if ωfk>bound
    ωf_conv = false
    @warn "Convergence not ensured: evaluation error of f too big: ωf = $(ωfk) > $bound = η_0*ΔTk "
  end
  if !ωf_int
    fk = obj(nlp,xk)
  end
  return fk, ωfk,ωf_conv
end

# Evaluate gradient and error, deal with the case gk = 0
function evaluate_grad_n_error(nlp,ck,ωg_int,ωg_fun)
  gk = nothing
  ωgk = nothing
  if ωg_int
    Ck = IntervalBox(ck)
    Gk = grad(nlp,Ck)
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
    ωgk =ωg_fun(ck)
    gk = grad(nlp,ck)
  end
  return gk,ωgk
end
