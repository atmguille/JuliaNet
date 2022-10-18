include("Neuron_pkg.jl")
include("Layer_pkg.jl")
include("NeuralNetwork_pkg.jl")
using .Neuron_pkg
using .Layer_pkg
using .NeuralNetwork_pkg

using ArgParse
using DelimitedFiles
using Statistics

include("ReadData.jl")
include("utils.jl")


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--input_file"
            help = "File with input values to train and test the network (modes 1 and 2). In mode 3, it is only the training file."
            required = true
        "--output_name"
            help = "Base name for the output files in which the predictions, ecm and hit rate are to be stored."
            required = true
        "--learning_rate"
            help = "Learning rate of the multilayer perceptron."
            arg_type = Float64
            required = true
        "--epochs"
            help = "Maximum number of epochs to perform training."
            arg_type = Int64
            required = true
        "--mode"
            help = "Mode of operation for reading data."
            arg_type = Int64
            required = true
        "--percentage"
            help = "Percentage of file data used in training. Exclusive of mode 1."
            arg_type = Float64
        "--input_test_file"
            help = "File with input values to test the network. Exclusive of mode 3."
        "--net_config"
            help = "List with the configuration of the hidden layers of the network. For example,
                    '[3,2,1]' indicates that the network will have 3 hidden layers, the first with 3 neurons,
                    the second with 2 neurons and the third with 1."
            arg_type = Vector{Int64}
            required = true
        "--normalize"
            help = "If specified, the input data is normalized."
            nargs = 0
    end

    return parse_args(s)
end

"""
    ECM(target::Vector, prediction::Vector)

Computes the mean square error (MSE) between the target and the prediction.
# Arguments:
- `target::Vector`: Target.
- `prediction::Vector`: Prediction

"""
function ECM(target::Vector{Float64}, prediction::Vector{Float64})
    return sum(map((x) -> x^2, prediction-target)) / size(prediction, 1)
end

"""
    predictions_acc_ECM(network::NeuralNetwork, inputs::Vector, outputs::Vector) -> (Vector, Float64, Float64)

Computes the predictions, the accuracy, the mean square error (MSE) and the confusion matrix of the neural network,
returning (predictions, accuracy, ECM, confusion_matrix).
# Arguments:
- `network::NeuralNetwork`: Neural network.
- `inputs::Vector`: Input values of the network
- `outputs::Vector`: Expected network output values

"""
function predictions_acc_ECM(network::NeuralNetwork_pkg.NeuralNetwork, inputs::Vector{Vector{Float64}},
                              outputs::Vector{Vector{Float64}})
    predictions = []
    n_inputs = size(inputs, 1)
    n_classes = size(outputs[1], 1)
    prediction_class = repeat([-1.], n_classes)
    ecm = 0
    confusion_matrix = zeros(Int64, (n_classes, n_classes))

    for i in 1:n_inputs
        attributes = inputs[i]
        classes = outputs[i]
        NeuralNetwork_pkg.Feedforward(network, attributes)
        prediction = [neuron.output_value for neuron in last(network.layers).neurons]
        ecm += ECM(classes, prediction)
        _, index_actual = findmax(classes)
        # The predicted class is the neuron that has been activated the most.
        _, index_pred = findmax(prediction)
        prediction_class[index_pred] = 1.
        confusion_matrix[index_pred, index_actual] += 1
        push!(predictions, copy(prediction_class))
        prediction_class[index_pred] = -1.
    end
    ecm /= n_inputs
    acc = sum(confusion_matrix[i, i] for i in 1:size(confusion_matrix, 1)) / n_inputs

    return predictions, acc, ecm, confusion_matrix
end


function main()

    parsed_args = parse_commandline()

    mode = parsed_args["mode"]

    ret = read_mode(mode, parsed_args)

    if ret == nothing
        return
    end

    inputs_training, outputs_training, inputs_test, outputs_test = ret

    output_name = parsed_args["output_name"]
    learning_rate = parsed_args["learning_rate"]
    epochs = parsed_args["epochs"]
    net_config = parsed_args["net_config"]
    normalize = parsed_args["normalize"]

    # The entries already contain the bias constant, then we subtract 1 for the number of attributes.
    num_attributes = size(inputs_training[1], 1) - 1
    num_classes = size(outputs_training[1], 1)

    # Initialize the network
    network = NeuralNetwork_pkg.CreateRandomNetwork([num_attributes; net_config; num_classes], -0.5, 0.5)
    # Normalize the input data if necessary
    if normalize
        # The function mean and std expect a different type than the one we process. We transform it with reduce
        mean_training = mean(reduce(hcat, inputs_training)', dims=1)
        std_training = std(reduce(hcat, inputs_training)', dims=1)
        # When including the bias, we must keep it at 1 despite normalize
        mean_training = [mean_training[1:num_attributes]; 0.]
        std_training = [std_training[1:num_attributes]; 1.]
        inputs_training = map((x) -> (x-mean_training) ./std_training, inputs_training)
        inputs_test = map((x) -> (x-mean_training) ./std_training, inputs_test)
    end

    #array_preds_test = []
    array_ecm_train = []
    array_acc_train = []
    array_ecm_test = []
    array_acc_test = []

    for epoch in 1:epochs
        for i in 1:size(inputs_training, 1)
            attributes = inputs_training[i]

            classes = outputs_training[i]
            NeuralNetwork_pkg.Feedforward(network, attributes)
            NeuralNetwork_pkg.Backpropagation(network, classes, learning_rate)
        end

        _, acc_train, ecm_train, conf_mat_train = predictions_acc_ECM(network, inputs_training, outputs_training)
        predictions_test, acc_test, ecm_test, conf_mat_test = predictions_acc_ECM(network, inputs_test, outputs_test)
        println("Epoch ", epoch)
        println("ECM Train: ", ecm_train, " ECM Test: ", ecm_test)
        println("Accuracy Train: ", acc_train, " Accuracy Test: ", acc_test)
        println("Confusion Train Array: ", conf_mat_train, " Confusion Test Array: ", conf_mat_test)
        #push!(array_preds_test, predictions_test)
        push!(array_ecm_train, ecm_train)
        push!(array_ecm_test, ecm_test)
        push!(array_acc_train, acc_train)
        push!(array_acc_test, acc_test)
    end

    #writedlm(output_name * "_pred_test.txt", array_preds_test)
    writedlm(output_name * "_ecm_train.txt", array_ecm_train)
    writedlm(output_name * "_ecm_test.txt", array_ecm_test)
    writedlm(output_name * "_acc_train.txt", array_acc_train)
    writedlm(output_name * "_acc_test.txt", array_acc_test)

    NeuralNetwork_pkg.Free(network)

end

main()
