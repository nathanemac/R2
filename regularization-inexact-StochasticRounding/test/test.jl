# using Knet
# include("../src/iterator.jl")

# # Define convolutional layer:
# struct Conv
#   w
#   b
#   f
# end
# (c::Conv)(x) = c.f.(pool(conv4(c.w, x) .+ c.b))
# Conv(w1, w2, cx, cy, f = relu) = Conv(param(w1, w2, cx, cy), param0(1, 1, cy, 1), f);

# # Define dense layer:
# struct Dense
#   w
#   b
#   f
# end
# (d::Dense)(x) = d.f.(d.w * mat(x) .+ d.b)
# Dense(i::Int, o::Int, f = relu) = Dense(param(o, i), param0(o), f);

# # Define a chain of layers:
# struct Chain
#   layers
#   Chain(args...) = new(args)
# end
# (c::Chain)(x) = (for l in c.layers
#   x = l(x)
# end;
# x)
# (c::Chain)(x, y) = nll(c(x), y)

# # Load MNIST data
# include(Knet.dir("data", "mnist.jl"))
# dtrn, dtst = mnistdata();

# # Train and test LeNet (about 30 secs on a gpu to reach 99% accuracy)
# LeNet1 = Chain(Conv(5, 5, 1, 20), Conv(5, 5, 20, 50), Dense(800, 500), Dense(500, 10, identity))
# adam!(LeNet1, dtrn)
# adam_accuracy = accuracy(LeNet1, dtst)
# @info "" adam_accuracy

# # Train and test LeNet (about 30 secs on a gpu to reach 99% accuracy)
# LeNet2 = Chain(Conv(5, 5, 1, 20), Conv(5, 5, 20, 50), Dense(800, 500), Dense(500, 10, identity))
# arig!(LeNet2, dtrn)
# accuracy(LeNet2, dtst)
# @info "" arig_accuracy
γ_1 = 0.33
γ_2 = 1.9
γ_3 = 15

if (!(0 ≤ γ_1 ≤ 1 ≤ γ_2 ≤ γ_3))
  error("you need the values to follow 0 ≤ γ_1 ≤1 ≤ γ_2 ≤ γ_3")
end

function γ(n, u)
  return n * u
end

function myforeach(f, n, u)
  print(f(n, u))
end
myforeach(γ, 2, 10)
