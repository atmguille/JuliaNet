using Random

"""
    read2(data_file::String) -> (Array{Float64}, Array{Float64})

Reads the file specified in Mode 2, returning (inputs, outputs).
Inputs contains a constant 1 at the end of each row, used as input for network bias.
# Arguments:
- `data_file`: name of the data file to read.

"""
function read2(data_file::String)
    file_lines = readlines(data_file)
    num_attributes, num_classes = split(file_lines[1])
    num_attributes, num_classes = parse(Int64, num_attributes), parse(Int64, num_classes)

    inputs = Vector{Vector{Float64}}()
    outputs = Vector{Vector{Float64}}()

    for line in file_lines[2:size(file_lines,1)]
        values = map((x) -> parse(Float64, x), split(line))
        # Add the 1 from the bias
        push!(inputs, [values[1:num_attributes]; [1]])
        # If outputs are binary, convert to bipolar
        push!(outputs, map((x) -> x == 0. ? -1. : x, values[num_attributes+1:size(values,1)]))
    end

    return inputs, outputs
end

"""
    read1(data_file::String, by::Float64) -> (Array{Float64}, Array{Float64}, Array{Float64}, Array{Float64})

Reads the file specified in Mode 1, returning
(input_training, output_training, input_test, output_test).
# Arguments:
- `data_file`: name of the data file to read.
- `percentage`: percentage of data for training

"""
function read1(data_file::String, percentage::Float64)
    inputs, outputs = read2(data_file)

    seed = round(Int64, time() * 1000)
    shuffle!(Random.seed!(seed), inputs)
    shuffle!(Random.seed!(seed), outputs)

    inputs_training = inputs[1:round(Int64, size(inputs,1)*percentage)]
    inputs_test = inputs[round(Int64, size(inputs,1)*percentage)+1:size(inputs,1)]
    outputs_training = outputs[1:round(Int64, size(outputs,1)*percentage)]
    outputs_test = outputs[round(Int64, size(outputs,1)*percentage)+1:size(outputs,1)]
    return inputs_training, outputs_training, inputs_test, outputs_test
end

"""
    read3(training_file::String, test_file::String) -> (Array{Float64},
                                                        Array{Float64},
                                                        Array{Float64},
                                                        Array{Float64})

Read the files specified in Mode 3, returning
(input_training, output_training, input_test, output_test).
# Arguments:
- `training_file`: name of the training data file.
- `test_file`: test data file name

"""
function read3(training_file::String, test_file::String)
    inputs_training, outputs_training = read2(training_file)
    inputs_test, outputs_test = read2(test_file)
    return inputs_training, outputs_training, inputs_test, outputs_test
end

"""
    read_mode(mode::Int64, parsed_args::Dict) -> (Array{Float64}, Array{Float64}, Array{Float64}, Array{Float64})

Reads the files included in parsed_args in the specified mode, returning
(input_training, output_training, input_test, output_test).
# Arguments:
- `mode`: read mode
- `parsed_args`: dictionary that must contain the name of the file to read in `input_file`.
                 If the mode requires it, it must contain the percentage in `percentage`.
                 If the mode requires it, it must contain the name of the test file in `input_test_file`.

"""
function read_mode(mode::Int64, parsed_args::Dict)
    input_file = parsed_args["input_file"]
    
    if mode == 1
        percentage = parsed_args["percentage"]
        if percentage == nothing
            println("It is necessary to indicate the percentage in mode 1.")
            return nothing
        end
        inputs_training, outputs_training, inputs_test, outputs_test = read1(input_file, percentage)
    elseif mode == 2
        inputs_training, outputs_training = read2(input_file)
        inputs_test, outputs_test = inputs_training, outputs_training
    elseif mode == 3
        input_test_file = parsed_args["input_test_file"]
        if input_test_file == nothing
            println("The file used for test in mode 3 must be specified.")
            return nothing
        end
        inputs_training, outputs_training, inputs_test, outputs_test = read3(input_file, input_test_file)
    end

    return inputs_training, outputs_training, inputs_test, outputs_test
end
