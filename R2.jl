using ADNLPModels
using NLPModels
using LinearAlgebra
using Plots
using ForwardDiff

# Definiton of the objective function and its gradient
Ros(x) = (1-x[1])^2 + (x[2]-x[1]^2)^2
#ΔRos(x)=[2(x[1]-1)-4x[1]*(x[2]-x[1]^2), 2(x[2]-x[1]^2)]


function R2(x,obj, maxiterations)

    ## This function takes as input an initial point and the objective function to minimize 
    ## and returns the vector of xk, the vector of f(xk) and the number of iterations

    #Initializing the constants of the algorithm
    η1=0.3
    η2=0.7  
    γ1=1/2
    γ2=2.0
    ϵ=10^-8

    # To store the vectors xk,fk and gk: 
    resxk=[]
    resf=[]
    resgk=[]

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
    while !((norm(gk)<ϵ) | (iter>maxiterations))
        sk=-gk/σk 
        if norm(sk)<eps()
            @warn "Stop because the step is lower than machine precision"
            break
        end

            #@show var to display one variable at every step if needed

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
        push!(resxk, xk)
        push!(resf,fk)
        push!(resgk,gk)
    end
    return(resxk,resf,iter,resgk)
end

# Storing the result for x,y to choose, obj and maxiter 
Res=R2([-2,2],Ros,1000)
IT=range(1,Res[3])

# x and y 
XK=Res[1]
XK1=[]
XK2=[]
for i=1:length(XK)
    push!(XK1, XK[i][1])
    push!(XK2,XK[i][2])
end

# Gradient of x and y
Gk=Res[4]
Gkx=[]
Gky=[]
for i=1:length(Gk)
    push!(Gkx, Gk[i][1])
    push!(Gky,Gk[i][2])
end



#pygui(true)

#To plot vectors x and y
plot(IT,XK1)
plot!(IT,XK2)


#To plot vectors Δx and Δy
plot(IT,Gkx)
plot!(IT,Gky)








## WITH NLPModels (Unfinished but not neccessary)


using OptimizationProblems, OptimizationProblems.ADNLPProblems
problems = setdiff(names(OptimizationProblems.ADNLPProblems), [:ADNLPProblems])
length(problems)
s=problems[end-1]
nlp=eval(s)()
nlp.meta.ncon

# Definiton of the objective function and its gradient

function R2NLP(x, nlp, maxiterations)

    ## This function takes as input an initial point and the objective function to minimize 
    ## and returns the vector of xk, the vector of f(xk) and the number of iterations

    #Initializing the constants of the algorithm
    η1=0.3
    η2=0.7  
    γ1=1/2
    γ2=2.0
    ϵ=10^-8

    # To store the vectors xk and fk: 
    resxk=[]
    resf=[]
    resgk=[]

    # Initializing the variables
    xk = copy(x)
    ρk=0
    ck=copy(xk)
    fk=obj(nlp,xk)
    gk=grad(nlp,xk)
    σk = 2^round(log2(norm(gk)+1))
    

    #T=obj(xk)+gk'*sk
    #n=T+norm(sk)^2/(σk^2)

    iter=1
    while !((norm(gk)<ϵ) | (iter>maxiterations))
        sk=-gk/σk 
        if norm(sk)<eps()
            @warn "Stop because the step is lower than machine precision"
            break
        end

        ck=xk.+sk
        ΔTk=-gk'*sk
        fck=obj(nlp,ck)
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
            gk=grad(nlp,xk)
        end

        iter+=1
        push!(resxk, xk)
        push!(resf,fk)
        push!(resgk,gk)
    end
    return(resxk, resf, iter,resgk)
end




# Storing the result for x,y to choose, obj and maxiter 
ResNLP=R2NLP(rand(100),nlp,1000) #Depends on the dimension of nlp
IT_NLP=range(1,Res[3])
XK_NLP=ResNLP[1]
Gk_NLP=ResNLP[4]
Fk_NLP=ResNLP[2]

for i=1:length(IT_NLP)
        plot!(IT_NLP[i],XK_NLP[i])
end





# x and y 

xk_NLP=[]
yk_NLP=[]
for i=1:length(XK)
    push!(xk_NLP, XK_NLP[i][1])
    push!(yk_NLP,XK_NLP[i][2])
end

# Gradient of x and y

Gkx_NLP=[]
Gky_NLP=[]
for i=1:length(Gk_NLP)
    push!(Gkx_NLP, Gk_NLP[i][1])
    push!(Gky_NLP,Gk_NLP[i][2])
end



#pygui(true)

#To plot vectors x and y
plot(IT_NLP,xk_NLP)
plot!(IT_NLP,yk_NLP)


#To plot vectors Δx and Δy
plot(IT_NLP,Gkx_NLP)
plot!(IT_NLP,Gky_NLP)