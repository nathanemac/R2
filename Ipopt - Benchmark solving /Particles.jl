using ADNLPModels, NLPModels, NLPModelsIpopt, DataFrames, LinearAlgebra, Distances, SolverCore, PyPlot

# Compute the potential of the n points
function Potential(x)
    v=reshape(x,3,N)'
    elts=pairwise(Euclidean(),v,dims=1)
    u_elts=[]
    for i=2:N-1
        for j=i+1:N
            push!(u_elts,1/elts[i,j])
        end
    end
    s=sum(u_elts)
    return(s)
end


# Define the constraints on these points (sum of the square of the coordinates = 1)
function constraints(x)
    c=[]
    for k=0:N-1
        push!(c, x[3k+1]^2 + x[3k+2]^2 + x[3k+3]^2)
    end
    return c
end


## Solving the problem

N=30
lcon=ucon=Float64.(ones(N))
z0=rand(N*3)

model=ADNLPModel(Potential, z0, constraints, lcon, ucon)
output= ipopt(model)
print(output)


# Plotting the solutions
x=[output.solution[3k+1] for k=0:N-1]
y=[output.solution[3k+2] for k=0:N-1]
z=[output.solution[3k+3] for k=0:N-1]

pygui(true)

fig=plt.figure(figsize=(20,20))
ax = plt.axes(projection ="3d")
 
ax.scatter3D(x, y, z)
plt.title("Répartition des charges dans l'espace")
 
plt.show()