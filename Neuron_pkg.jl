module Connection_pkg

export Connection

abstract type AbstractNeuron end

mutable struct Connection{T<:AbstractNeuron}
    weight::Float64
    previous_weight::Float64
    value::Float64
    neuron::T
end

"""
    Create(weight::Float64, neuron::Neuron) -> Connection.

# Arguments:

- `weight::Float64`: Weight of the connection.
- `neuron::Neuron`: Neuron to connect to

"""
Create(weight::Float64, neuron::AbstractNeuron) = Connection(weight, 0.0, 0.0, neuron)


"""
    Free(connection::Connection).

# Arguments:

- `connection::Connection`: Connection to free.
"""
function Free(connection::Connection)
    connection = nothing
    GC.gc()
end

"""
    Propagate(connection::Connection, value::Float64)

# Arguments:

- `connection::Connection`: Connection to propagate the value through.
- value::Float64`: Value to propagate between neurons through the connection.
"""
function Propagate(connection::Connection, value::Float64)
    connection.neuron.input_value += value * connection.weight
    connection.value = value
end

end

module Neuron_pkg

using ..Connection_pkg

export Neuron

@enum Type begin
    Direct
    Bias
    McCulloch
    Perceptron
    Adaline
    Sigmoid
end


mutable struct Neuron <: Connection_pkg.AbstractNeuron
    type::Type
    threshold::Float64
    input_value::Float64
    output_value::Float64
    connections::Vector{Connection}
end

"""
    Create(threshold::Float64, type::Type) -> Neuron.

# Arguments:

- `threshold::Float64`: Activation threshold of the neuron.
- `type::Type`: Type of the neuron (Direct, Bias, McCulloch, Perceptron, Adaline or Sigmoid)

"""
Create(threshold::Float64, type::Type) = Neuron(type, threshold, 0.0, 0.0, Vector{Connection}())

"""
    Free(neuron::Neuron).

Frees the neuron as well as the connections coming out of it.
# Arguments

- `neuron::Neuron`: Neuron to release

"""
function Free(neuron::Neuron)
    for connection in neuron.connections
        Connection_pkg.Free(connection)
    end
    neuron = nothing
    GC.gc()
end

"""
    Initialize(neuron::Neuron, x::Float64)

# Arguments:
- `neuron::Neuron`: Neuron to initialize.
- `x::Float64`: Input value of the neuron

"""
Initialize(neuron::Neuron, x::Float64) = neuron.input_value = x

"""
    Connect(source_neuron::Neuron, target_neuron::Neuron, weight::Float64)

A connection is created between the specified neurons, adding it to the source neuron's connection list
# Arguments:
- `source_neuron::Neuron`: Source neuron of the connection.
- `target_neuron::Neuron`: Target neuron of connection
- `weight::Float64`: Connection weight

"""
Connect(source_neuron::Neuron, target_neuron::Neuron, weight::Float64) = push!(source_neuron.connections,
                                                                                   Connection_pkg.Create(weight, target_neuron))

"""
    __sigmoid(x::Float64)

Sigmoid function
# Arguments:
- `x::Float64`: Value to evaluate.

"""
__sigmoid(x::Float64) = 2.0 / (1.0 + exp(-x)) - 1.0

"""
    Trigger(neuron::Neuron)

Processes the input value of the neuron with the corresponding activation according to its type.
# Arguments:
- `neuron::Neuron`: Neuron to trigger.

"""
function Trigger(neuron::Neuron)
    if neuron.type == Direct
        neuron.output_value = neuron.input_value
    elseif neuron.type == Bias
        neuron.output_value = 1.0
    elseif neuron.type == McCulloch
        neuron.output_value = neuron.input_value >= neuron.threshold ? 1.0 : 0.0
    elseif neuron.type == Perceptron
        if neuron.input_value > neuron.threshold
            neuron.output_value = 1.0
        elseif neuron.input_value < -1*neuron.threshold
            neuron.output_value = -1.0
        else
            neuron.output_value = 0.0
        end
    elseif neuron.type == Adaline
        if neuron.input_value >= 0
            neuron.output_value = 1.0
        else
            neuron.output_value = -1.0
        end
    elseif neuron.type == Sigmoid
        neuron.output_value = __sigmoid(neuron.input_value)
    end
end

"""
    Propagate(neuron::Neuron)

Propagates the output value of the neuron through its connections.
# Arguments:
- `neuron::Neuron`: Neuron to propagate.

"""
function Propagate(neuron::Neuron)
    for connection in neuron.connections
        Connection_pkg.Propagate(connection, neuron.output_value)
    end
end

function DerivativeActivation(neuron::Neuron)
    if neuron.type == Sigmoid
        sigmoid_eval = __sigmoid(neuron.input_value)
        return 0.5 * (1 + sigmoid_eval) * (1 - sigmoid_eval)
    else
        println("[!] Derivative not defined for this neuron type")
        return 0.0
    end
end

end
