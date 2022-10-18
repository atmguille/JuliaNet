module Layer_pkg

using ..Neuron_pkg

export Layer

mutable struct Layer
    neurons::Vector{Neuron}
end

"""
    Create() -> Layer

Creates an empty layer.

"""
Create() = Layer(Vector{Neuron}())

"""
    Free(Layer::Layer)

Frees the layer and the neurons it contains.
# Arguments:
- `layer::Layer`: Layer to free.

"""
function Free(layer::Layer)
    for neuron in layer.neurons
        Neuron_pkg.Free(neuron)
    end
    layer = nothing
    GC.gc()
end

"""
    Initialize(layer::Layer)

Initializes the layer's neurons to input value 0.
# Arguments:
- `layer::Layer`: Layer to initialize.

"""
function Initialize(layer::Layer)
    for neuron in layer.neurons
        Neuron_pkg.Initialize(neuron, 0.0)
    end
end

"""
    Add(layer::Layer, neuron::Neuron)

Adds a neuron to the layer
# Arguments:
- `layer::Layer`: Layer to add the neuron to.
- `neuron::Neuron`: Neuron to add

"""
function Add(layer::Layer, neuron::Neuron)
    push!(layer.neurons, neuron)
end

"""
    Connect(layer::Layer, neuron::Neuron, min_weight::Float64, max_weight::Float64)

Connects all neurons in the specified layer, to the specified neuron, with random weights between min_weight and max_weight.
# Arguments:
- `layer::Layer`: Output layer.
- `neuron::Neuron`: Input neuron
- `min_weight::Float64`: Minimum weight
- `max_weight::Float64`: Maximum weight

"""
function Connect(layer::Layer, neuron::Neuron, min_weight::Float64, max_weight::Float64)
    for source_neuron in layer.neurons
        weight = rand() * (max_weight - min_weight) + min_weight
        Neuron_pkg.Connect(source_neuron, neuron, weight)
    end
end

"""
    Connect(layer::Layer, next_layer::Layer, min_weight::Float64, max_weight::Float64)

Connects all neurons between the indicated layers, with random weights between min_weight and max_weight.
# Arguments:
- `layer::Layer`: Output layer.
- `next_layer::Layer`: Input layer
- `min_weight::Float64`: Minimum weight
- `max_weigth::Float64`: Maximum weight

"""
function Connect(layer::Layer, next_layer::Layer, min_weight::Float64, max_weight::Float64)
    for target_neuron in next_layer.neurons
        if target_neuron.type != Neuron_pkg.Bias
            Connect(layer, target_neuron, min_weight, max_weight)
        end
    end
end

"""
    Trigger(layer::Layer).

Triggers all neurons in the layer.
# Arguments:
- `layer::Layer`: Layer to trigger

"""
function Trigger(layer::Layer)
    for neuron in layer.neurons
        Neuron_pkg.Trigger(neuron)
    end
end

"""
    Propagate(layer::Layer)

Propagate all neurons in the layer
# Arguments:
- `layer::Layer`: Layer to propagate

"""
function Propagate(layer::Layer)
    for neuron in layer.neurons
        Neuron_pkg.Propagate(neuron)
    end
end

end

