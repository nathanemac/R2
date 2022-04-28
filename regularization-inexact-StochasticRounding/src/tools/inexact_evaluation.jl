"""
Evaluate artificial inexact objectif value.
 - ω_f : user-specified error threshold.
"""
function inexact_obj(nlp::AbstractNLPModel, x::Vector, ω_f::AbstractFloat)
  u = 2 * rand() - 1
  NLPModels.increment!(nlp, :neval_obj)
  f = obj(nlp, x)

  return f, f + u * ω_f
end

"""
Evaluate artificial inexact gradient.
 - ω_g : user-specified error threshold.
"""
function inexact_grad(nlp::AbstractNLPModel, x::Vector, ω_g::AbstractFloat)
  n = nlp.meta.nvar
  u = 2 * rand(n) .- 1
  u = u / norm(u)

  NLPModels.increment!(nlp, :neval_grad)
  ∇f = grad(nlp, x)
  ∇f_norm = BLAS.nrm2(n, ∇f, 1)

  λ = ω_g / (1 - ω_g) * ∇f_norm
  return ∇f, ∇f + λ * u
end
