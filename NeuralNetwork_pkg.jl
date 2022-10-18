module NeuralNetwork_pkg

using ..Layer_pkg
using ..Neuron_pkg

mutable struct NeuralNetwork
    layers::Vector{Layer}
end

"""
    Create() -> NeuralNetwork

Creates an empty neural network

"""
Create() = NeuralNetwork(Vector{Layer}())

"""
    Free(neural_network::NeuralNetwork)

Frees the neural network and all the layers it contains.
# Arguments:
- `neuronal_network::NeuronalNetwork`: Neuronal network to free.

"""
function Free(neural_network::NeuralNetwork)
    for layer in neural_network.layers
        Layer_pkg.Free(layer)
    end
    red_neuronal = nothing
    GC.gc()
end

"""
    Initialize(neural_network::NeuralNetwork)

Initializes all neurons in the neural network to 0.
# Arguments:
- `neural_network::NeuralNetwork`: Neural network to initialize.

"""
function Initialize(neural_network::NeuralNetwork)
    for layer in neural_network.layers
        Layer_pkg.Initialize(layer)
    end
end

"""
    Add(neural_network::NeuralNetwork, layer::Layer)

Adds a layer to the neural network
# Arguments:
- `neural_network::NeuralNetwork`: Neural network to add layer to.
- `layer::Layer`: Layer to be added

"""
function Add(neural_network::NeuralNetwork, layer::Layer)
    push!(neural_network.layers, layer)
end

"""
    Trigger(neuronal_network::NeuronalNetwork)

Triggers all layers of the neural network.
# Arguments:
- `neuronal_network::NeuronalNetwork`: Neural network to be triggered

"""
function Trigger(neural_network::NeuralNetwork)
    for layer in neural_network.layers
        Layer_pkg.Trigger(layer)
    end
end

"""
    Propagate(neural_network::NeuralNetwork)

Propagate all layers of the neural network.
# Arguments:
- `neuronal_network::NeuronalNetwork`: Neural network to propagate

"""
function Propagate(neural_network::NeuralNetwork)
    for layer in neural_network.layers
        Layer_pkg.Propagate(layer)
    end
end

"""
    Propagate_and_Trigger(neural_network::NeuralNetwork)

Successively propagate and trigger all layers of the neural network.
# Arguments:
- `neural_network::NeuralNetwork`: Neural network to propagate and fire.

"""
function Propagate_and_Trigger(neural_network::NeuralNetwork)
    for index_layer in 1:(size(neural_network.layers,1)-1)
        Layer_pkg.Propagate(neural_network.layers[index_layer])
        Layer_pkg.Trigger(neural_network.layers[index_layer + 1])
    end
end

"""
    CreateRandomNetwork(configuration::Vector{Int64}, min_weight::Float64, max_weight::Float64)

Creates a random multilayer neural network based on a configuration.
# Arguments:
- `configuration::Vector{Int64}`: list of integers that indicate the number of neurons
                                  in each layer
- `min_weight::Float64`: minimum value of the weights
- `max_weight::Float64`: maximum value of weights

"""
function CreateRandomNetwork(configuration::Vector{Int64}, min_weight::Float64, max_weight::Float64)
    neural_network = Create()
    n_layers = size(configuration, 1)
    for (layer_index, n_neurons) in enumerate(configuration)
        layer = Layer_pkg.Create()

        # The first layer has no activation function
        if layer_index == 1
            for i in 1:n_neurons
                neuron = Neuron_pkg.Create(0.0, Neuron_pkg.Direct)
                Layer_pkg.Add(layer, neuron)
            end
        else
            for i in 1:n_neurons
                neuron = Neuron_pkg.Create(0.0, Neuron_pkg.Sigmoid)
                Layer_pkg.Add(layer, neuron)
            end
        end

        # The last layer is unbiased
        if layer_index != n_layers
            Layer_pkg.Add(layer, Neuron_pkg.Create(1.0, Neuron_pkg.Bias))
        end
        # The first layer is the only one with no connection to the previous layer
        if layer_index != 1
            Layer_pkg.Connect(last(neural_network.layers), layer, min_weight, max_weight)
        end
        Add(neural_network, layer)
    end

    return neural_network
end

"""
    Feedforward(neural_network::NeuralNetwork, input_values::Vector).

Feedforward one cycle of the neural network. It consists of:
* Initialize the input layer to the input values.
* Trigger the input layer
* Unloading the neurons from the neural network (initialization of input values to 0)
* Successively propagate and fire all layers (and their neurons) of the neural network.

# Arguments:
- `neural_network::NeuralNetwork`: Neural network
- `input_values::Vector`: Input values

"""
function Feedforward(neural_network::NeuralNetwork, input_values::Vector{Float64})
    input_layer = neural_network.layers[1]
    for i in 1:size(input_values, 1)
        Neuron_pkg.Initialize(input_layer.neurons[i], input_values[i])
    end
    Layer_pkg.Trigger(input_layer)
    Initialize(neural_network)
    Propagate_and_Trigger(neural_network)
end

"""
    Backpropagation(network::NeuralNetwork, true_classes::Vector, learning_rate::Float64).

Updates all neural network weights using the backpropagation method.

# Arguments:
- `neural_network::NeuralNetwork`: Neural network.
- `true_classes::Vector`: True classes of the input data
- `learning_rate::Float64`: Learning rate

"""
function Backpropagation(neural_network::NeuralNetwork, true_classes::Vector{Float64},
                         learning_rate::Float64)
    output_layer = last(neural_network.layers)
    deltas = Vector{Float64}()

    # deltas of the output layer
    for i in 1:size(true_classes, 1)
        delta = true_classes[i] - output_layer.neurons[i].output_value
        delta *= Neuron_pkg.DerivativeActivation(output_layer.neurons[i])
        push!(deltas, delta)
    end

    n_layers = size(neural_network.layers, 1)
    # Iterate layers from back to front
    for (layer_index, layer) in enumerate(reverse(neural_network.layers[1:n_layers - 1]))
       deltas_layer = Vector{Float64}()
       layer_outputs = []
        for neuron in layer.neurons
            push!(layer_outputs, neuron.output_value)
            # It is not necessary to calculate delta for Bias or for the first layer of the network.
            if neuron.type == Neuron_pkg.Bias || layer_index == n_layers - 1
                continue
            end
            neuron_weights = []
            for connection in neuron.connections
                push!(neuron_weights, connection.weight)
            end
            delta = Neuron_pkg.DerivativeActivation(neuron) * (deltas' * neuron_weights)
            push!(deltas_layer, delta)
        end
        # update layer weights
        delta_weights = learning_rate .* (layer_outputs * deltas')
        for (neuron_index, neuron) in enumerate(layer.neurons)
            for (connection_index, connection) in enumerate(neuron.connections)
                connection.previous_weight = connection.weight
                connection.weight += delta_weights[neuron_index, connection_index]
            end
        end
        deltas = deltas_layer
        end
    end


end
