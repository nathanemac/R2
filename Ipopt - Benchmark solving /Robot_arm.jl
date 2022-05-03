using JuMP, ADNLPModels, NLPModelsIpopt, NLPModels, Ipopt

N=99 # number of intervals
n=N+1 # number of timesteps
L=4.5 #total length of the robot arm
# h = 

Np=(0:1:N)
Nv=(0.5:1:N-0.5)
Na=(1:1:N-1)    

ρ0=L
ρN=L
θ0=0
θN=2π/3
ϕ0=π/4
ϕN=π/4
ρ_dot0, θ_dot0, ϕ_dot0, ρ_dotN, θ_dotN, ϕ_dotN, ρ_accN, θ_accN, ϕ_accN  = zeros(9)



# X : vector of variables, of the form : [ρ(t=t1); ρ(t=t2); ... ρ(t=tf), θ(t=t1), ..., then ρ_dot, ..., then ρ_acc, .. ϕ_acc, tf]
# There are N+1 values of each 9 variables 
# X = [ρ, θ, ϕ, ρ_dot, θ_dot, ϕ_dot, ρ_acc, θ_acc, ϕ_acc, tf]

model=Model()
set_optimizer(model, Ipopt.Optimizer)


# initializing X
X0=zeros(9n+1)
X0[1]=L
X0[n+1]=0
X0[2n+1]=π/4
X0[end]=1.0

lb_ρ=zeros(n)
ub_ρ=ones(n)*L

lb_θ=ones(n)*π
ub_θ=-ones(n)*π

lb_ϕ=zeros(n)
ub_ϕ=ones(n)*π

lb_ρ_acc=-ones(n)/L
ub_ρ_acc=ones(n)/L

# Declaring the variables 
@variable(model, 0 <= tf)
@variable(model, lb_ρ[i] <= ρ[i=1:n] <= ub_ρ[i])    
@variable(model, lb_θ[i] <= θ[i=1:n] <= ub_θ[i])
@variable(model, lb_ϕ[i] <= ϕ[i=1:n] <= ub_ϕ[i])
@variable(model, ρ_dot[i=1:n]) 
@variable(model, θ_dot[i=1:n])
@variable(model, ϕ_dot[i=1:n])
@variable(model, lb_ρ_acc[i] <= ρ_acc[i=1:n] <= ub_ρ_acc[i])
@variable(model, θ_acc[i=1:n])
@variable(model, ϕ_acc[i=1:n])

# Declaring the constraints on the variables
    # constraints on ρ, θ, ϕ
@constraint(model, c_ρ1[i=1:N], ρ[i+1] <= ρ[i] + ρ_dot[i]*tf/N)
@constraint(model, c_ρ2[i=1:N], ρ[i+1] >= ρ[i] + ρ_dot[i]*tf/N)
@constraint(model, c_θ1[i=1:N], θ[i+1] <= θ[i] + θ_dot[i]*tf/N)
@constraint(model, c_θ2[i=1:N], θ[i+1] >= θ[i] + θ_dot[i]*tf/N)
@constraint(model, c_ϕ1[i=1:N], ϕ[i+1] <= ϕ[i] + ϕ_dot[i]*tf/N)
@constraint(model, c_ϕ2[i=1:N], ϕ[i+1] >= ϕ[i] + ϕ_dot[i]*tf/N)

# Todo finir contraintes sur ρ, theta et phi_dot

    # constraints on inertia
        # Iθ
@NLconstraint(model, c_θ_acc1[i=1:n], θ_acc[i] <= 1/(((L-ρ[i])^3+ρ[i]^3)/3*sin(ϕ[i]^2)))
@NLconstraint(model, c_θ_acc2[i=1:n], θ_acc[i] >= -1/(((L-ρ[i])^3+ρ[i]^3)/3*sin(ϕ[i]^2)))

        # Iϕ todo



# todo creer fobj et mettre en contrainte les valeurs finales des variables









function objective(X)
    X[end]
end

function c(X)
    tf=X[end]
    c_ρ=[]
    for i=1:N
        push!(c_ρ, X[i+1] = X[i] + X[3n+i]*tf/N











    #functions for positions

function ρ(t)
    rρ=[]
    for i in Np 
        push!(rρ,((N-t)*ρ0 + t*ρN)/N)
    end
    return rρ
    end

function θ(t)
    rθ=[]
    for i in Np 
        push!(rθ,((N-t)*θ0 + t*θN)/N)
    end
    return rθ
    end

function ϕ(t)
    rϕ=[]
    for i in Np 
        push!(rϕ,((N-t)*ϕ0 + t*ϕN)/N)
    end
    return rϕ
    end


    # functions for velocities

function ρ_dot(t)
    ρ_d = N*(ρ(t+0.5) - ρ(t-0.5))/t
    end

function θ_dot(t)
    θ_d = N*(θ(t+0.5) - θ(t-0.5))/t
    end

function ϕ_dot(t)
    ϕ_d =  N*(ϕ(t+0.5) - ϕ(t-0.5))/t
    end


    #functions for accelerations

function ρ_acc(t)
    ρ_a = N*(ρ_dot(t+0.5) - ρ_dot(t-0.5))/t
    end


function θ_acc(t)
    θ_a = N*(θ_dot(t+0.5) - θ_dot(t-0.5))/t
    end

function ϕ_acc(t)
    ϕ_a = N*(ϕ_dot(t+0.5) - ϕ_dot(t-0.5))/t
    end


    #functions for controls

function uρ(t)
    return L*ρ_acc(t)
end

function uθ(t)
    return Iθ(t).*θ_acc(t)
end

function uϕ(t)
    return Iϕ(t).*ϕ_acc(t)
end



    #functions for inertia moments

function Iθ(t)
    return ((L .- ρ(t)).^3 .+ ρ(t).^3)/3 .* sin.(ϕ(t)).^2
end

function Iϕ(t)
    return ((L .- ρ(t)).^3 .+ ρ(t).^3)/3
end



    # x=[ρ, θ, ϕ, uρ, uθ, uϕ]

# upper and lower constraints
function c(t)
    return [ρ(t); θ(t); ϕ(t); uρ(t); uθ(t); uϕ(t)]
end

lcon=zeros(6*N)
for i=1:N
    lcon[i]=L
    lcon[N+i]=π
    lcon[2N+i]=π
    lcon[3N+i]=1
    lcon[4N+i]=1
    lcon[5N+i]=1
end

ucon=zeros(6*N)
for i=1:N
    ucon[i]=0
    ucon[N+i]=-π
    ucon[2N+i]=0
    ucon[3N+i]=-1
    ucon[4N+i]=-1
    ucon[5N+i]=-1
end

function objective(x)
    return 



