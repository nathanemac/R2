N=8
L=4.5

Np=(0:1:N)
Nv=(0.5:1:N-0.5)
Na=(1:1:N-1)    

ρ0=L
ρN=L
θ0=0
θN=2π/3
ϕ0=π/4
ϕN=π/4
ρ_dot0, θ_dot0, ϕ_dot0, ρ_dotN, θ_dotN, ϕ_dotN = zeros(6)



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