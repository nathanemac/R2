" 
    Solves the optimization problem min(f(x)) with quadratic regularization
    Input: (x::Vector, f::function to minimize, maxiterations::Int, η1,η2,γ1,γ2,ϵ_abs,ϵ_rel :: Float64, verbose::Bool )
    Output: (GenericExecutionStats)
"

function R2(nlp::AbstractNLPModel{T, S}, 
    kwargs_dict = Dict(kwargs...),
    x0 = pop!(kwargs_dict, :x0, nlp.meta.x0),
    xk, k, outdict = R2(x -> obj(nlp, x), args..., x0; kwargs_dict...), 
    kwargs_dict...) where {T, S}

  return GenericExecutionStats(
    outdict[:status],
    nlp,
    solution = xk,
    objective = outdict[:fk],
    dual_feas = outdict[:norm_gk],
    elapsed_time = outdict[:elapsed_time],
    iter = k,
    solver_specific = Dict(
        :Fhist => outdict[:Fhist]
      ),
    )
end


function R2(
    nlp::AbstractNLPModel{T, S},
    options::ROSolverOptions,
    x0::AbstractVector,) where {T, S}

    start_time = time()
    elapsed_time = 0.0
    ϵ = options.ϵ
    verbose::Bool = false
    MaxIterations = options.MaxIterations
    MaxTime = options.MaxTime

    σmin = T(options.σmin),
    η1 = T(options.η1),
    η2 = T(options.η2),
    γ1 = T(options.γ1),
    γ2 = T(options.γ2),


    # Initialisation
    iter=0
    xk = copy(x0)
    Fobj_hist = zeros(maxIter)
    ρk = T(0)
    fk = obj(nlp, xk)
    gk = similar(xk)
    grad!(nlp, xk, gk)
    norm_gk=norm(gk)
    σk = 2^round(log2(norm(gk) + 1)) # The closest exact-computed power of 2 from gk

    # Stopping criterion: 
    ϵ = ϵ_abs + ϵ_rel*norm_gk
    optimal = norm_gk ≤ ϵ
    tired = iter > MaxIterations

    if verbose
        @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "σ"
        infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e" iter fk norm_gk σk
    end

    status = :unknown
    ck = similar(xk)
    
    while !(optimal | tired)

        Fobj_hist[k] = fk
        ck .= xk .- (gk./σk)
        ΔTk= norm_gk^2/ σk
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
            grad!(nlp, xk, gk)
            norm_gk = norm(gk)
        end


        iter += 1
        optimal = norm_gk ≤ ϵ
        tired = iter > MaxIterations | elapsed_time > MaxTime
        elapsed_time = time() - start_time

        if verbose
            #infoline *= @sprintf "  %8.1e  %7.1e" ρk elapsed_time
            @info infoline
            infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e" iter fk norm_gk σk
        end

    end
    
    status = if optimal
        :first_order
      elseif tired
        :max_iter
      else
        :exception
      end

      outdict = Dict(
        :Fhist => Fobj_hist[1:iter],
        :status => status,
        :xk => xk
        :fk => fk,
        :dual_feas = norm_gk,
        :elapsed_time => elapsed_time,
        :iter => iter
      )

    return xk, iter, outdict
end