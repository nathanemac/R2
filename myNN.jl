using LinearAlgebra, ForwardDiff, Printf, NLPModels, ADNLPModels, SolverCore, PyPlot, Plots, NLPModelsIpopt
include("R2_vf.jl")

function activation(x, W, b)
    # Evaluates sigmoid function on input x, weights W and bias b
    n=length(W[:,1])
    y=[]
    for i=1:n
        push!(y, 1/(1 + exp(-(W[i,:]'*x+b[i]))))
    end
    return y
end

function cost(A)
    # gives the cost of the set of weights and bias
    w1 = A[1:10]
    W1 = reshape(w1, 5, 2)

    w2 = A[11:30]
    W2 = reshape(w2, 4, 5)

    W3 = A[31:34]'

    b1 = A[35:39]
    b2 = A[40:43]
    b3 = A[44]

    cost_vec = similar(A,a) # size of X, vector of inputs
    for j=1:a
        x=XX[:,j]
        a1 = activation(x, W1, b1)     
        a2 = activation(a1, W2, b2)     
        a3 = activation(a2, W3, b3)
        cost_vec[j] = norm(y[j].*y[j] .- a3.*a3)
    end
    cost_value = norm(cost_vec,2) 
    return cost_value
end

function curve(A,B,C,xx)
    l=length(xx)
    return A .* xx.^2 .+ B.* xx .+ C.*ones(l)
end

function range_curve(A,B,C,xx)
    range = zeros(length(xx))
    l = length(xx)
    for i = 1:l
        range[i] += curve(A,B,C,xx[i])[1]
    end
    min = minimum(range)
    max = maximum(range)
    return min,max
end

# I chose to build a fully-connected - two hidden layers network. 
# I want, from a set of coordinates, to predict whose category the point belongs. 

function create_train_dataset(a)
    # create the data set to train the network
    xundertrain = [] # will contain the points under the curve y=2x-1
    xuppertrain = [] # will contain the points upper the curve y=2x-1
    Xtrainx = rand(start:stop,a)'
    Xtrainy = rand(range_curve(A,B,C,xx)[1]-10:range_curve(A,B,C,xx)[2]+10,a)'
    Xtrain = [Xtrainx;
            Xtrainy]

    for i=1:a
        (Xtrain[:,i][2] - curve(A, B, C, Xtrain[:,i][1])[1] < 0) ? push!(xundertrain, Xtrain[:,i]) : push!(xuppertrain, Xtrain[:,i])
    end

    X = vcat(xundertrain, xuppertrain)
    y = vcat(zeros(length(xundertrain)), ones(length(xuppertrain)))

    XX = zeros(2, a)
    for i=1:a
        XX[1,i]+=X[i][1]
        XX[2,i]+=X[i][2]
    end

    return (XX,y)
end

function train_network(a)
    # Initializing weights and biases
    X0 = rand(44)
    nlp = ADNLPModel(cost, X0)
    opti = ipopt(nlp)

    # Defining the optimal parameters 
    Xop = opti.solution
    w1o = Xop[1:10]
    W1o = reshape(w1o, 5, 2)

    w2o = Xop[11:30]
    W2o = reshape(w2o, 4, 5)

    W3o = Xop[31:34]'

    b1o = Xop[35:39]
    b2o = Xop[40:43]
    b3o = Xop[44]

    return W1o,W2o,W3o,b1o,b2o,b3o
end

########################################################

function NN(x)
    # Compute the output y from input X
    a1 = activation(x, W1o, b1o)     
    a2 = activation(a1, W2o, b2o)     
    a3 = activation(a2, W3o, b3o)
    output = a3[1]
    round(output)
end


# creation d'une database aleatoire pour tester le reseau
function create_test_dataset(b)

    xundert = [] # will contain the points under the curve y=2x-1
    xuppert = [] # will contain the points upper the curve y=2x-1
    Xtestx = rand(start:stop,b)'
    Xtesty = rand(range_curve(A,B,C,xx)[1]-10:range_curve(A,B,C,xx)[2]+10,b)'
    Xtest = [Xtestx;
            Xtesty]

    for i=1:b
        ((Xtest[:,i][2] - curve(A, B, C, Xtest[:,i][1])[1]) < 0) ? push!(xundert, Xtest[:,i]) : push!(xuppert, Xtest[:,i])
    end

    Xundert = zeros(2, length(xundert))
    Xuppert = zeros(2, length(xuppert))
    for i=1:length(xundert)
        Xundert[1,i]+=xundert[i][1]
        Xundert[2,i]+=xundert[i][2]
    end
    for i=1:length(xuppert)
        Xuppert[1,i]+=xuppert[i][1]
        Xuppert[2,i]+=xuppert[i][2]
    end


    yundert = []
    yuppert = []
    for i=1:b
        (NN(Xtest[:,i]) == 0) ? push!(yundert, Xtest[:,i]) : push!(yuppert, Xtest[:,i])
    end
    Yundert = zeros(2,length(yundert))
    Yuppert = zeros(2,length(yuppert))

    for i=1:length(yundert)
        Yundert[1,i]+=yundert[i][1]
        Yundert[2,i]+=yundert[i][2]
    end
    for i=1:length(yuppert)
        Yuppert[1,i]+=yuppert[i][1]
        Yuppert[2,i]+=yuppert[i][2]
    end
    
    return Xundert,Xuppert,Yundert,Yuppert, xundert, xuppert
end

function stats(Xundert, Xuppert, Yundert, Yuppert, xundert, xuppert)
    accuracy_under = 0
    total_under = length(Xundert[1,:])
    wrong_under=[]
    for i=1:length(Yundert[1,:])
        if Yundert[:,i] in xundert
            accuracy_under+=1
        else
            push!(wrong_under,Yundert[:,i])
        end
    end

    accuracy_upper = 0
    total_upper = length(Xuppert[1,:])
    wrong_upper=[]
    for i=1:length(Yuppert[1,:])
        if Yuppert[:,i] in xuppert
            accuracy_upper+=1
        else
            push!(wrong_upper,Yuppert[:,i])
        end
    end

    Wrong_under = zeros(2,length(wrong_under))
    for i=1:length(wrong_under)
        Wrong_under[1,i] += wrong_under[i][1]
        Wrong_under[2,i] += wrong_under[i][2]
    end
    Wrong_upper = zeros(2,length(wrong_upper))
    for i=1:length(wrong_upper)
        Wrong_upper[1,i] += wrong_upper[i][1]
        Wrong_upper[2,i] += wrong_upper[i][2]
    end

    return Wrong_under, accuracy_under, Wrong_upper, accuracy_upper
end

###############################################

a = 50 # size of the train set
b = 550 # size of the test set
A,B,C = -2,2,2
start,stop,ϵ = 0, 15.0, 1e-4
xx = start:ϵ:stop

XX, y = create_train_dataset(a)
W1o,W2o,W3o,b1o,b2o,b3o = train_network(a)
Xundert, Xuppert, Yundert, Yuppert, xundert, xuppert = create_test_dataset(b)


###############################################

# now we have 4 vectors : 
# Xundert, Xuppert : contain the points up the curve and down the curve
# Yundert, Yuppert : contain the points predicted as up the curve and predicted as down the curve


statistics = stats(Xundert, Xuppert, Yundert, Yuppert, xundert, xuppert)
Wrong_under = statistics[1]
Wrong_upper = statistics[3]

accuracy = (statistics[2]+statistics[4])/b

xx = start:ϵ:stop
Plots.plot(xx, curve(A,B,C,xx), label = "curve")
scatter!(Xundert[1,:], Xundert[2,:], label="under the curve")
scatter!(Xuppert[1,:], Xuppert[2,:], label="upper the curve")
scatter!(Wrong_upper[1,:], Wrong_upper[2,:], label = "wrong upper")
scatter!(Wrong_under[1,:], Wrong_under[2,:], label = "wrong under")