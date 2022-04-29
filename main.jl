# 1 bring the libraries
using LinearAlgebra
using ForwardDiff
using Printf
using NLPModels
using ADNLPModels
using SolverCore

include("R2_DNN.jl")

# 2 define a NLP (Problem ) aka #Ros(x) = (1-x[1])^2 + (x[2]-x[1]^2)^2 
nlp = ADNLPModel(x -> (1 - x[1])^2 + (x[2] - x[1]^2)^2, [-1, 2.0])

# Optional turn on/off the table 
#verbose=true


# 3 select a solver aka R2 and send the problem to it 

stats = R2(
    nlp,
    10000,
    0.3,
    0.7,
    1/2,
    2.0,
    1e-6,
    1e-6,
    false
)
print(stats)
