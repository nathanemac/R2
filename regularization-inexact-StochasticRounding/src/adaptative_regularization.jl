"""
Perform adaptative regularization optimization with exact gradient
"""
# function norm_est() #from matlab --> write it effient as possible 
# end

function adaptative_regularization(
  nlp;
  # σ_0 = 50.0, # intial σ # TODO can we change this ? #Todo we can use the norm of second order derivative 
  σ_min = 1.0, # TODO why do we need this intial value?  set to smaller 
  η_1 = 1e-4,
  η_2 = 0.75,
  γ_1 = 0.33, # TODO check 0 ≤ γ_1 ≤1 ≤ γ_2 ≤ γ_3
  γ_2 = 1,
  γ_3 = 5,
  ϵ_abs = 1e-6,
  ϵ_rel = 1e-6,
  maxiter = 10000, #TODO 500
  verbose::Bool = false,
)
  # Initialization
  n = nlp.meta.nvar
  iter = 0

  # Evaluate the objective fonction and its derivative at the initial point
  x = copy(nlp.meta.x0) # TODO it takes initial x, let the user pass inital x, if empty ... error promt 
  # λ, ϕ  = eigs(hess_op(helper_nlp, copy(helper_nlp.meta.x0)), which=:LM) # norm of ∇2f(x_0) getting the largest Eigenvalue 
  σ_0, _ = normest(hess_op(nlp, copy(nlp.meta.x0))) #TODO gives us ERror with
  # σ_0 = 50
  verbose && @info "" nlp.meta.name nlp.meta.nvar σ_0 σ_min η_1 η_2 γ_1 γ_2 γ_3
  σ = σ_0
  #TODO check this float 64 , check for float32
  x_trial = similar(x)
  fx = obj(nlp, x)
  ∇fx = grad(nlp, x)
  ∇fx_norm = norm(∇fx)

  # Define the stopping criterion
  ϵ = ϵ_abs + ϵ_rel * ∇fx_norm
  n
  if verbose
    @info @sprintf "%5s  %9s  %7s  %7s  %8s  %7s" "iter" "f" "‖∇f‖" "σ" "ρ" "time"
    infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e" iter fx ∇fx_norm σ
  end
  # Stopping crterion
  optimal = ∇fx_norm ≤ ϵ
  tired = iter > maxiter
  start_time = time()
  elapsed_time = 0.0

  while !(optimal | tired)
    # Step calculation
    x_trial .= x .- (1 / σ) .* ∇fx #TODO x_k+1 =xk+sk, Question step_k=- (1 / σ) .* ∇fx is , what is the expersion of M_k()
    # M(s)'=0 , s = -(1/σ) ∇fx
    # f(x+s)= f(x) + ∇fx*s   (scalar product)  # print the s, x to examine
    # M(s) = = f(x) + ∇fx*s  
    # grad(M,s) = 0 +  ∇fx

    # M(s)= f(x) + ∇fx*s +(σ/2)( sTs) quadratic model 
    #grad(M,s) = 0 +∇fx  + σ* s =0  # sigma helps us estimate second order derivayive of f
    # s = -∇fx/σ

    f_trial = obj(nlp, x_trial)

    # Regularisation parameter update
    # σ is used in calculation of step and ρ 
    ρ = σ * (fx - f_trial) / ∇fx_norm^2 #TODO 2\sigma?
    if ρ < η_1
      σ = γ_3 * σ
    elseif ρ ≥ η_2
      σ = max(σ_min, γ_1 * σ)
    end

    # Acceptance of the trial point (Accepting the step)
    if ρ ≥ η_1
      x .= x_trial
      fx = f_trial
      grad!(nlp, x, ∇fx)
      ∇fx_norm = norm(∇fx)
    end

    iter += 1
    optimal = ∇fx_norm ≤ ϵ
    tired = iter > maxiter

    elapsed_time += time() - start_time
    if verbose
      infoline *= @sprintf "  %8.1e  %7.1e" ρ elapsed_time
      @info infoline
      infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e" iter fx ∇fx_norm σ
    end
  end

  if verbose
    @info infoline
  end

  status = if optimal
    :first_order
  elseif tired
    :max_iter
  else
    :exception
  end

  return GenericExecutionStats(
    status,
    nlp,
    solution = x,
    objective = fx,
    dual_feas = ∇fx_norm,
    elapsed_time = elapsed_time,
    iter = iter,
  )
end
