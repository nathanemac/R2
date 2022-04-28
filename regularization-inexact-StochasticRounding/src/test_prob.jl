using StochasticRounding
using NLPModels
using ADNLPModels
using OptimizationProblems.ADNLPProblems
using Arpack
using LinearAlgebra
using TSVD

T = Float32sr
nlp = arglina(type = Val(T))
x = copy(nlp.meta.x0)
dummy2 = hess_op(nlp, x)
# dummy2.eltype(Float32)
promote_type(eltype(dummy2), Float32)
dummy2
convert(Vector{float(eltype(dummy2))}, dummy2)
# dummy2= hess_op(nlp, Float32.(x))
# dummy2.type(Float32)
# λ, ϕ  = eigs(dummy2, which=:LM)
tsvd(dummy2)
