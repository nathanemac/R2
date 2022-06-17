
"""
Author: D. Monnet 
Quadratic regularization algorithm
"""

function quadratic_regularization(
  nlp;
  σ_min=nothing,
  η_1 = 0.3,
  η_2 = 0.7,
  γ_1 = 2^(-1),
  γ_2 = 2.0,
  x0=nothing,
  maxiter = 1000,
  verbose::Bool = false,
  log::Bool = false
)



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

  ϵ_abs = (eps(type))^(1 / 3)
  ϵ_rel = (eps(type))^(1 / 3)

  # Initialization
  iter = 0
  xk = copy(x0)
  
  verbose && @info "" nlp.meta.name nlp.meta.nvar σ_min η_1 η_2 γ_1 γ_2

  # Check the conditions and send error 

  0 < γ_1 || error("γ_1 must be > 0")
  0 < γ_2 || error("γ_2 must be > 0")
  γ_1 ≤ 1 || error("γ_1 must be <= 1")
  γ_1 < 1 < γ_2 || error("0 < γ_1 < 1 < γ_2 ")
  0 < η_1 ≤ η_2 < 1 || error("must be 0 <η_1 ≤ η_2 < 1")

  #conversion needed to ensure all computations are run in initial precision
  γ_1 =  convert(type,γ_1)
  γ_2 =  convert(type,γ_2)
  
  ρk = 0

  # Evaluate the objective fonction and its derivative at the initial point
  ck = copy(xk)
  fk = obj(nlp, xk)
  gk = grad(nlp,xk)
  gk_norm = norm(gk)
  #σ0 = gk_norm+1 # ensures ||s_0|| ≈ 1, avoid big steps at first iterations. add eps to avoid division by 0
  σ_0 = 2^round(log2(gk_norm+1)) # ensures ||s_0|| ≈ 1 with sigma_0 = 2^n with n an interger, i.e. sigma_0 is exactly representable in n bits 
  σk = convert(type,σ_0)
  
  # print @info header
  if verbose
    infoline = @sprintf "%5s  %9s  %9s  %7s  %7s  %7s  %9s  %9s  " "iter" "fk" "f(ck)" "‖gk‖" "σk" "ρk" "||sk||" "||xk||\n"
    @info infoline
    if log
      open("output.txt","a") do io
        write(io, infoline)
      end
    end
    #infoline = @sprintf "%5d  %9.2e %9.2e  %7.1e  %7.1e  %7.1e  %7.1e" iter fk f_trial gk_norm σk μk ρk
  end
  # Stopping criteria
  ϵ = ϵ_abs + ϵ_rel * gk_norm
  optimal = gk_norm ≤ ϵ 
  tired = iter > maxiter

  status = nothing
  start_time = time()
  elapsed_time = 0.0

  while !(optimal | tired)
    # Step calculation
    sk = convert.(type,-gk./σk)
    if norm(sk) <eps(type)
      @warn "Algo stops because step is lower than machine precision"
      status = :small_step
      break
    end
    ck = xk .+ sk
    
    # Compute approximated taylor series decrease
    ΔTk= -gk' * sk

    f_trial = obj(nlp,ck)
    if f_trial == -Inf
      status = :unbounded
      break
    end
    
    if(fk==f_trial)
      #@show Fk Ftrial ϕk norm(xk) norm(sk) ωgk μk obj(nlp,xk) obj(nlp,ck) ΔTk xk==ck xk sk ck
      @warn "fk==f_trial"
     # error("fk==f_trial")
    end
    ρk = (fk - f_trial)/ΔTk 

    
    if ρk ≥ η_2 
      σk = max(σ_min,  σk*γ_1)
    elseif ρk < η_1
      σk = σk * γ_2
    end


    if verbose
      infoline = @sprintf "%5d  %9.2e %9.2e  %7.1e  %7.1e  %7.1e  %9.2e  %9.2e\n" iter fk f_trial gk_norm σk ρk norm(sk) norm(xk)
        if log
          open("output.txt","a") do io
          write(io, infoline)
          end
        end
        @info infoline
    end

    # Acceptance of the trial point (Accepting the step)
    if ρk ≥ η_1
      xk = ck
      fk = f_trial
      gk = grad(nlp,xk)
      gk_norm = norm(gk)
    end

    iter += 1
    optimal = gk_norm ≤ ϵ
    tired = iter > maxiter    
  end
  
  elapsed_time = time() - start_time
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