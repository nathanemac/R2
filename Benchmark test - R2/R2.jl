" 
    Solves the optimization problem min(f(x)) with quadratic regularization
    Input: (x::Vector, f::function to minimize, maxiterations::Int, η1,η2,γ1,γ2,ϵ_abs,ϵ_rel :: Float64, verbose::Bool )
    Output: (GenericExecutionStats)
"
function R2(
            nlp;
            σ_min = nothing,
            maxiterations=1000,
            η1 = 0.3,
            η2 = 0.7,
            γ1 = 1 / 2,
            γ2 = 2.0,
            ϵ_abs = 1e-6,
            ϵ_rel = 1e-6,
            verbose::Bool = false)

    # Initializing the variables
    type = typeof(nlp.meta.x0[1])
    x0 = copy(nlp.meta.x0)
    x0 = convert.(type,x0)

    if σ_min === nothing
        σ_min = eps(type)
    else 
        σ_min = convert(type,σ_min)
    end

    # Initialisation
    iter=0
    xk = copy(x0)

    ρk = 0
    ck = copy(xk)
    fk = obj(nlp, xk)
    gk = grad(nlp, xk)
    norm_gk=norm(gk)
    σk = 2^round(log2(norm(gk) + 1)) # The closest exact-computed power of 2 from gk

    γ1 =  convert.(type,γ1)
    γ2 =  convert.(type,γ2)
    σk =  convert.(type,σk)



    # Stopping criterion: 
    ϵ = ϵ_abs + ϵ_rel*norm_gk
    optimal = norm_gk ≤ ϵ
    tired = iter > maxiterations

    if verbose
        @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "σ"
        infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e" iter fk norm_gk σk
    end

    start_time = time()
    elapsed_time = 0.0

    while !(optimal | tired)

        sk = convert.(type,-gk./σk)

        if norm(sk) < eps(Float16)
            @warn "Stop because the step is lower than machine precision"
            status = :small_step
            break
        end 

        ck = xk .+ sk
        ΔTk = -gk' * sk
        fck = obj(nlp, ck)
        if fck == -Inf
            status = :unbounded
            break
        end

        ρk = (fk - fck) / ΔTk


        # Recomputing if conditions on ρk not reached
        if ρk >= η2
            σk = max(σ_min, γ1 * σk)

        elseif ρk < η1
            σk = σk * γ2

        end

        # Acceptance of the new candidate
        if ρk >= η1
            xk = ck
            fk = fck
            gk = grad(nlp, xk)
            norm_gk=norm(gk)
        end


        iter += 1
        optimal = norm_gk ≤ ϵ
        tired = iter > maxiterations

        elapsed_time += time() - start_time

        if verbose
            #infoline *= @sprintf "  %8.1e  %7.1e" ρk elapsed_time
            @info infoline
            infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e" iter fk norm_gk σk
        end

        global status

        status = if optimal
            :first_order
          elseif tired
            :max_iter
          else
            :exception
          end
    end
    
    return GenericExecutionStats(
            status,
            nlp,
            solution = xk,
            objective = fk,
            dual_feas = norm_gk,
            elapsed_time = elapsed_time,
            iter = iter,
            )
end
