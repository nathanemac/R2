
using StochasticRounding, BenchmarkTools, BFloat16s
A = rand(Float32, 1000);
B = rand(Float32, 1000);
A_sr, B_sr = Float32sr.(A), Float32sr.(B);
# A_sr = rand(Float32sr,1000); #TODO I get this error  # we could create a wrapper 
# B_sr = rand(Float32sr,1000);
print(typeof(A_sr))
@btime +($A, $B)
@btime +($A_sr, $B_sr)
