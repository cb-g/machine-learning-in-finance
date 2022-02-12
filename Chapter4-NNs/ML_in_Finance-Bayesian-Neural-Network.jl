using Plots, DataFrames, Flux, Plots, Random, ReverseDiff
using Turing # v0.20

# create half-moon data:
function make_moons(n_samples::Int = 100)
    
    n = n_samples

    n_samples_out = trunc(Int, round(n_samples / 2))
    n_samples_in = n_samples - n_samples_out

    outer_circ_x = cos.(LinRange(0, π, n_samples_out))
    outer_circ_y = sin.(LinRange(0, π, n_samples_out))
    inner_circ_x = 1 .- cos.(LinRange(0, π, n_samples_in))
    inner_circ_y = 1 .- sin.(LinRange(0, π, n_samples_in)) .- 0.5

    X = hcat(vcat(outer_circ_x, inner_circ_x), vcat(outer_circ_y, inner_circ_y))
    Y = hcat(zeros(Int, 1, 2 .* n_samples_out), ones(Int, 1, 2 .* n_samples_in))

    return X, Y, n

end

X, Y, n = make_moons()

# rearrange half-moon data:
n_2 = trunc(Int, n/2)
xscatter_0 = X[1:n_2]
xscatter_1 = X[1+n_2:n]
yscatter_0 = X[1+n:n+n_2]
yscatter_1 = X[1+n+n_2:end]

# plot half-moon data:
plot(
    xscatter_0, yscatter_0, 
    seriestype = :scatter, 
    legend = true,
    label = "Class 0",
    title = "half-moons problem"
)
plot!(
    xscatter_1, yscatter_1, 
    seriestype = :scatter, 
    legend = true,
    label = "Class 1"
)

# optional: half-moon data in a dataframe:
half_moon_df = DataFrame(
    X = vcat(xscatter_0, xscatter_1),
    Y = vcat(yscatter_0, yscatter_1),
    Class = vcat(zeros(Int, n_2), ones(Int, n_2))
)

# Turing settings:
Turing.setprogress!(true)
Turing.setadbackend(:reversediff)

# architecture:
network_shape = [
    (5,2, :tanh),
    (5,5, :tanh),
    (2,5, :σ)]

# Regularization, parameter variance, and total number of parameters.
alpha = 0.09
sig = sqrt(1.0 / alpha)
num_params = sum([i * o + i for (i, o, _) in network_shape])

# This modification of the unpack function generates a series of vectors given a network shape.
function unpack(θ::AbstractVector, network_shape::AbstractVector)
    index = 1
    weights = []
    biases = []
    for layer in network_shape
        rows, cols, _ = layer
        size = rows * cols
        last_index_w = size + index - 1
        last_index_b = last_index_w + rows
        push!(weights, reshape(θ[index:last_index_w], rows, cols))
        push!(biases, reshape(θ[last_index_w+1:last_index_b], rows))
        index = last_index_b + 1
    end
    return weights, biases
end

# Generate an abstract neural network given a shape, and return a prediction.
function nn_forward(x, θ::AbstractVector, network_shape::AbstractVector)
    weights, biases = unpack(θ, network_shape)
    layers = []
    for i in eachindex(network_shape)
        push!(layers, Dense(weights[i],
            biases[i],
            eval(network_shape[i][3])))
    end
    nn = Chain(layers...)
    return nn(x)
end

# rearrange half-moon data yet again:
xscatter = vcat(xscatter_0, xscatter_1)
yscatter = vcat(yscatter_0, yscatter_1)
xs = Array([[xscatter[i]; yscatter[i]] for i = 1:n])
ts = vcat(zeros(Int, n_2), ones(Int, n_2))

# General Turing specification for a BNN model.
@model bayes_nn_general(xs, ts, network_shape, num_params) = begin
    θ ~ MvNormal(zeros(num_params), sig .* ones(num_params))
    preds = nn_forward(xs, θ, network_shape)
    for i = 1:length(ts)
        ts[i] ~ Bernoulli(preds[i])
    end
end

# Perform inference.
num_samples = 2000
ch2 = sample(bayes_nn_general(hcat(xs...), ts, network_shape, num_params), NUTS(0.65), num_samples);

# This function makes predictions based on network shape.
function nn_predict(x, theta, num, network_shape)
    mean([nn_forward(x, theta[i,:], network_shape)[1] for i in 1:10:num])
end;

# Extract the θ parameters from the sampled chain.
params2 = MCMCChains.group(ch2, :θ).value

# classification plot
plot(
    xscatter_0, yscatter_0, 
    seriestype = :scatter, 
    legend = true,
    label = "Class 0",
    title = "half-moons problem"
)
plot!(
    xscatter_1, yscatter_1, 
    seriestype = :scatter, 
    legend = true,
    label = "Class 1"
)
x_range = collect(range(-2.5,stop=3.5,length=200))
y_range = collect(range(-1.2,stop=1.7,length=200))
Z = [nn_predict([x, y], params2, length(ch2), network_shape)[1] for x=x_range, y=y_range]
contour!(x_range, y_range, Z)
