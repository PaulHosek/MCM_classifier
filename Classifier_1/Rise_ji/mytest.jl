

using GraphicalModelLearning



# if abspath(PROGRAM_FILE) == @__FILE__
model = FactorGraph([0.0 0.1 0.2; 0.1 0.0 0.3; 0.2 0.3 0.0])
samples = sample(model, 100000)
learned = learn(samples)

err = abs.(convert(Array{Float64,2}, model) - learned)
println(err)

