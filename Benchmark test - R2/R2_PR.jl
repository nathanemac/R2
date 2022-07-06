mutable struct R2Solver{T, V, M <: AbstractNLPModel{T, V}} 
x::V
gx::V
end

function R2Solver(nlp::M;) where {T, V, M <: AbstractNLPModel{T, V}}
    nvar = nlp.meta.nvar
    x = V(undef, nvar)
    gx = V(undef, nvar)
    return R2Solver{T, V, M}(x, gx)
end


@doc (@doc R2Solver) function R2(
  nlp::AbstractNLPModel;
  x::V = nlp.meta.x0,
  kwargs...,
) where {V}
  solver = R2Solver(nlp;)
  return solve!(solver, nlp; xk = x, kwargs...)
end

function solve!(
    solver::R2Solver{T,V},
    nlp::AbstractNLPModel{T, V};
    xk::V = copy(nlp.meta.x0),
    ϵ_abs::T = eps(T)^(1 / 3),
    ϵ_rel::T = eps(T)^(1 / 3),
    η1 = T(0.3),
    η2 = T(0.7),
    γ1 = T(1/2),
    γ2 = 1/γ1,
    σmin = eps(T),
    MaxTime::Float64 = 30.0,
    MaxIterations::Int = 100,
    verbose::Bool = false,
  ) where {T, V}

  if !unconstrained(nlp)
    error("R2 should only be called for unconstrained problems.")
  end

  start_time = time()
  elapsed_time = 0.0

  xk = solver.x
  ∇xk = solver.gx

  iter = Int(0)
  ρk = T(0)
  fk = obj(nlp, xk)
  
  grad!(nlp, xk, ∇xk)
  norm_∇xk=norm(∇xk)
  σk = 2^round(log2(norm(∇xk) + 1)) # The closest exact-computed power of 2 from ∇xk

    # Stopping criterion: 
  ϵ = ϵ_abs + ϵ_rel*norm_∇xk
  optimal = norm_∇xk ≤ ϵ
  tired = (iter > MaxIterations) | (elapsed_time > MaxTime)

  if verbose
      @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "σ"
      infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e" iter fk norm_∇xk σk
  end

  status = :unknown
  start_time = time()
  elapsed_time = 0.0
  ck = similar(xk)
    
  while !(optimal | tired)

    ck .= xk .- (∇xk./σk)
    ΔTk= norm_∇xk^2/ σk
    fck = obj(nlp, ck)
    if fck == Inf
        status = :unbounded
        break
    end

    ρk = (fk - fck) / ΔTk 


        # Recomputing if conditions on ρk not reached
    if ρk >= η2
        σk = max(σmin, γ1 * σk)
    elseif ρk < η1
        σk = σk * γ2
    end

        # Acceptance of the new candidate
    if ρk >= η1
        xk .= ck
        fk = fck
        grad!(nlp, xk, ∇xk)
        norm_∇xk = norm(∇xk)
    end


    iter += 1
    optimal = norm_∇xk ≤ ϵ
    tired = iter > MaxIterations

    elapsed_time = time() - start_time

    if verbose
        @info infoline
        infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e" iter fk norm_∇xk σk
    end

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
      solution = xk,
      objective = fk,
      dual_feas = norm_∇xk,
      elapsed_time = elapsed_time,
      iter = iter,
    )
end