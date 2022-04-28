import Base:
  length,
  size,
  tail,
  iterate,
  eltype,
  IteratorSize,
  IteratorEltype,
  haslength,
  SizeUnknown,
  @propagate_inbounds,
  HasEltype
using LinearAlgebra

# UPDATE
mutable struct ARIG
  σ_0::AbstractFloat
  σ_min::AbstractFloat
  η_1::AbstractFloat
  η_2::AbstractFloat
  γ_1::AbstractFloat
  γ_3::AbstractFloat
  γ_2::AbstractFloat
  iter::Int
  σ::AbstractFloat
  τ::AbstractFloat
end

ARIG(; σ_0 = 50.0, σ_min = 1.0, η_1 = 1e-4, η_2 = 0.75, γ_1 = 0.33, γ_2 = 1.0, γ_3 = 5.0) =
  ARIG(σ_0, σ_min, η_1, η_2, γ_1, γ_2, γ_3, 0, σ_0, 1 / σ_0)
arig(
  f,
  d;
  σ_0 = 50.0,
  σ_min = 1.0,
  η_1 = 1e-4,
  η_2 = 0.75,
  γ_1 = 0.33,
  γ_2 = 1.0,
  γ_3 = 5.0,
  kwargs...,
) = minimize(f, d, ARIG(σ_0, σ_min, η_1, η_2, γ_1, γ_2, γ_3, 0, σ_0, 1 / σ_0); kwargs...)
arig!(args...; kwargs...) =
  for y in arig(args...; kwargs...)
  end

clone(a::ARIG) = ARIG(a.σ_0, a.σ_min, a.η_1, a.η_2, a.γ_1, a.γ_2, a.γ_3, 0, a.σ_0, 1 / a.σ_0)

function trial_step(g::AbstractArray, p::ARIG)
  # Step calculation
  s = -1 / p.σ * g
  return s
end

update!(x::Param, f, g) = update!(x.value, f, g, x.opt)

function update!(
  x::AbstractArray,
  f::AbstractFloat,
  f_trial::AbstractFloat,
  g::AbstractArray,
  p::ARIG,
)
  ρ = p.σ * (f - f_trial) / norm(g)^2

  # Acceptance of the trial point
  (ρ ≥ p.η_1) && axpy!(-1 / p.σ, g, x)

  # Regularisation parameter update
  if ρ < p.η_1
    p.σ = p.γ_3 * p.σ
  elseif ρ ≥ p.η_2
    p.σ = max(p.σ_min, p.γ_1 * p.σ)
  end
end

# TRAIN

minimize(f, d::I, a = ARIG(); params = nothing) where {I} =
  Minimize{I}(d, f, a, params, typeof(f(first(d)...)))
minimize!(x...; o...) =
  for x in minimize(x...; o...)
  end

struct Minimize{I}
  data::I
  func
  algo
  params
  eltype
end

length(m::Minimize) = length(m.data)
size(m::Minimize) = size(m.data)
eltype(m::Minimize) = m.eltype
IteratorSize(::Type{Minimize{I}}) where {I} = IteratorSize(I)
IteratorEltype(::Type{<:Minimize}) = Base.HasEltype()

function iterate(m::Minimize, s...)
  @info "enter my iterate"
  next = iterate(m.data, s...)
  next === nothing && return nothing
  (args, s) = next
  y = @diff m.func(args...)
  for x in (m.params === nothing ? params(y) : m.params)
    if x.opt === nothing
      x.opt = clone(m.algo)
    end
    if isa(a, ARIG)
      d = trial_step(grad(x, y), x.opt)
      y_trial = @diff m.func(value(x) .+ d)
      update!(x, value(y), value(y_trial), grad(y, x))
    else
      update!(x, grad(y, x))
    end
  end
  return (value(y), s)
end
