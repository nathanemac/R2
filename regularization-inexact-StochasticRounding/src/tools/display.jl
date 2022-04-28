export display_level_curves, display_algorithm_behaviour

# display_level_curves(args...; kwargs...) = error("Plots is required for display")
# display_algorithm_behaviour(args...; kwargs...) = error("Plots is required for display")

"""
    display_level_curves(nlp::AbstractNLPModel, size::Integer, xmin::AbstractFloat, xmax::AbstractFloat, ymin::AbstractFloat, ymax::AbstractFloat)

Outputs level curves for problem `nlp`.

#### Arguments
* `nlp::AbstractNLPModel`: the problem
* `size:Integer`: number of discretization points per sides, generates a size*size mesh grid
* `xmin,xmax,ymin,ymax::Float`: delimit the borders of the mesh

#### Return value
* an object Plots.Plot{Plots.GRBackend}
"""
function display_level_curves(
  nlp::AbstractNLPModel,
  size::Integer,
  xmin::AbstractFloat,
  xmax::AbstractFloat,
  ymin::AbstractFloat,
  ymax::AbstractFloat,
)
  name = nlp.meta.name
  xlen, ylen = abs(xmax - xmin), abs(ymax - ymin)
  I = xmin:(xlen / (size - 1)):xmax
  J = ymin:(ylen / (size - 1)):ymax
  Z = [obj(nlp, [i, j]) for i in I, j in J]
  plt = plot(I, J, Z', fill = false, title = "Optimization Problem: $name", nlevels = 1000)
  return plt
end

"""
    display_algorithm_behaviour()

Outputs one or several execution tracks on top of the problem nlp level curves.

#### Arguments
* nlp::AbstractNLPModel : the problem solved
* meshsize::Integer : number of discretization points per sides, generates a size*size mesh grid
* iterates::Array{AbstractFloat}... : one or several exectution traces

#### Return value
* an object Plots.Plot{Plots.GRBackend}
"""
function display_algorithm_behaviour(
  nlp::AbstractNLPModel,
  meshsize::Integer,
  iterates::Vector{Array{AbstractFloat, 2}},
  labels::Vector{String},
)
  x = hcat(iterates...)
  xmin, ymin = min(x[1, :]...), min(x[2, :]...)
  xmax, ymax = max(x[1, :]...), max(x[2, :]...)
  plt = display_level_curves(nlp, meshsize, xmin, xmax, ymin, ymax)
  for k = 1:length(iterates)
    x1, x2 = [iterates[k][1, :]], [iterates[k][2, :]]
    plot!(plt, x1, x2, label = labels[k], marker = :circle)
  end
  return plt
end
