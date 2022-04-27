#Ros(x) = (1-x[1])^2 + (x[2]-x[1]^2)^2 ---- If test needed on a basic function
function R2_DNN(x, obj, maxiterations)

    ## This function takes as input an initial point and the objective function to minimize 
    ## and returns x_opti, f(x_opti) and the number of iterations

    #Initializing the constants of the algorithm
    η1=0.3
    η2=0.7  
    γ1=1/2
    γ2=2.0
    ϵ=10^-8

    # Initializing the variables
    xk = copy(x)
    ρk=0
    ck=copy(xk)
    fk=obj(xk)
    gk=ForwardDiff.gradient(obj,xk)
    σk = 2^round(log2(norm(gk)+1)) # La puissance de 2 plus proche de la norme de gk
    

    #T=obj(xk)+gk'*sk
    #m=T+norm(sk)^2*σk*1/2

    iter=0
    while !((norm(gk)<ϵ) | (iter>maxiterations)) #rajouter erreur relative
        sk=-gk/σk 
        if norm(sk)<eps()
            @warn "Stop because the step is lower than machine precision"
            break
        end
        
        ck=xk.+sk
        ΔTk=-gk'*sk
        fck=obj(ck)
        ρk=(fk-fck)/ΔTk

        # Recomputing if conditions on ρk not achieved
        if ρk>=η2
            σk=γ1*σk

        elseif ρk<η1
            σk=σk*γ2

        end

        # Acceptance of the new candidate
        if ρk >= η1
            xk=ck
            fk=fck
            gk=ForwardDiff.gradient(obj,xk)
        end

        iter+=1
    end
    return(xk, fk, iter)
end