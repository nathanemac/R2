using JuMP, ADNLPModels, NLPModelsIpopt, NLPModels, Ipopt

N=99 # number of intervals
n=N+1 # number of timesteps
L=4.5 #total length of the robot arm

# X : vector of variables, of the form : [ρ(t=t1); ρ(t=t2); ... ρ(t=tf), θ(t=t1), ..., then ρ_dot, ..., then ρ_acc, .. ϕ_acc, tf]
# There are N+1 values of each 9 variables 
# X = [ρ, θ, ϕ, ρ_dot, θ_dot, ϕ_dot, ρ_acc, θ_acc, ϕ_acc, tf]

model=Model()
set_optimizer(model, Ipopt.Optimizer)


#bounds

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

    # constrains on initial time and final time

        #on positions
@constraint(model, c_ρ_0_1, ρ[1] <= L)
@constraint(model, c_ρ_0_2, ρ[1] >= L)
@constraint(model, c_ρ_f_1, ρ[end] <= L)
@constraint(model, c_ρ_f_2, ρ[end] >= L)

@constraint(model, c_θ_0_1, θ[1] <= 0)
@constraint(model, c_θ_0_2, θ[1] >= 0)
@constraint(model, c_θ_f_1, θ[end] <= 2π/3)
@constraint(model, c_θ_f_2, θ[end] >= 2π/3)

@constraint(model, c_ϕ_0_1, ϕ[1] <= π/4)
@constraint(model, c_ϕ_0_2, ϕ[1] >= π/4)
@constraint(model, c_ϕ_f_1, ϕ[end] <= π/4)
@constraint(model, c_ϕ_f_2, ϕ[end] >= π/4)

        #on speeds
@constraint(model, c_ρ_dot_0_1, ρ_dot[1] <= 0)
@constraint(model, c_ρ_dot_0_2, ρ_dot[1] >= 0)
@constraint(model, c_ρ_dot_f_1, ρ_dot[end] <= 0)
@constraint(model, c_ρ_dot_f_2, ρ_dot[end] >= 0)

@constraint(model, c_θ_dot_0_1, θ_dot[1] <= 0)
@constraint(model, c_θ_dot_0_2, θ_dot[1] >= 0)
@constraint(model, c_θ_dot_f_1, θ_dot[end] <= 0)
@constraint(model, c_θ_dot_f_2, θ_dot[end] >= 0)

@constraint(model, c_ϕ_dot_0_1, ϕ_dot[1] <= 0)
@constraint(model, c_ϕ_dot_0_2, ϕ_dot[1] >= 0)
@constraint(model, c_ϕ_dot_f_1, ϕ_dot[end] <= 0)
@constraint(model, c_ϕ_dot_f_2, ϕ_dot[end] >= 0)

        #on accelerations 
@constraint(model, c_ρ_acc_0_1, ρ_acc[1] <= 0)
@constraint(model, c_ρ_acc_0_2, ρ_acc[1] >= 0)
@constraint(model, c_ρ_acc_f_1, ρ_acc[end] <= 0)
@constraint(model, c_ρ_acc_f_2, ρ_acc[end] >= 0)

@constraint(model, c_θ_acc_0_1, θ_acc[1] <= 0)
@constraint(model, c_θ_acc_0_2, θ_acc[1] >= 0)
@constraint(model, c_θ_acc_f_1, θ_acc[end] <= 0)
@constraint(model, c_θ_acc_f_2, θ_acc[end] >= 0)

@constraint(model, c_ϕ_acc_0_1, ϕ_acc[1] <= 0)
@constraint(model, c_ϕ_acc_0_2, ϕ_acc[1] >= 0)
@constraint(model, c_ϕ_acc_f_1, ϕ_acc[end] <= 0)
@constraint(model, c_ϕ_acc_f_2, ϕ_acc[end] >= 0)


    # constraints on ρ_dot, θ_dot, ϕ_dot

@constraint(model, c_ρ_dot1[i=1:N], ρ_dot[i+1] <= ρ_dot[i] + ρ_acc[i]*tf/N)
@constraint(model, c_ρ_dot2[i=1:N], ρ_dot[i+1] >= ρ_dot[i] + ρ_acc[i]*tf/N)
@constraint(model, c_θ_dot1[i=1:N], θ_dot[i+1] <= θ_dot[i] + θ_acc[i]*tf/N)
@constraint(model, c_θ_dot2[i=1:N], θ_dot[i+1] >= θ_dot[i] + θ_acc[i]*tf/N)
@constraint(model, c_ϕ_dot1[i=1:N], ϕ_dot[i+1] <= ϕ_dot[i] + ϕ_acc[i]*tf/N)
@constraint(model, c_ϕ_dot2[i=1:N], ϕ_dot[i+1] >= ϕ_dot[i] + ϕ_acc[i]*tf/N)

    # constraints on inertia
        # Iθ
@NLconstraint(model, c_θ_acc1[i=1:n], θ_acc[i] <= 1/(((L-ρ[i])^3+ρ[i]^3)/3*sin(ϕ[i]^2)))
@NLconstraint(model, c_θ_acc2[i=1:n], θ_acc[i] >= -1/(((L-ρ[i])^3+ρ[i]^3)/3*sin(ϕ[i]^2)))

        # Iϕ todo
@NLconstraint(model, c_ϕ_acc1[i=1:n], ϕ_acc[i] <= ((L-ρ[i])^3+ρ[i]^3)/3)
@NLconstraint(model, c_ϕ_acc2[i=1:n], ϕ_acc[i] >= -((L-ρ[i])^3+ρ[i]^3)/3)



# create the objective function

@objective(model, Min, tf)

optimize!(model)
print(model)











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



