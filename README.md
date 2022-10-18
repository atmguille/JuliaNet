# JuliaNet

This repository contains a educational implementation of a generic deep neural network in Julia. Its purpose is to facilitate the understanding of how deep neural networks work, both for the feed-forward and the back-propagation. It comes from a more specific version initially developed for the Neurocomputation course at UAM, which imposed the structure of the library for the neural network (check it [here](https://github.com/AlejandroSantorum/Apuntes_Mat_IngInf/blob/master/5_Curso/Neurocomputaci%C3%B3n/Pr%C3%A1cticas/Pr%C3%A1ctica1/Enunciado_Pr%C3%A1ctica1.pdf)). Due to this structural imposition for educational purposes, we could not make the library as efficient as possible (via matrix multiplications instead of for loops). However, it is still very efficient thanks to Julia, since it is **10 times faster** than the Python implementation that the rest of the students did.

## Prerequisites

### Install Julia

You can download and install Julia in this [link](https://julialang.org/downloads/).

### Install ArgParse

Once Julia has been installed, you have to install the ArgParse package via the following command:

```bash
julia -e 'using Pkg; Pkg.add("ArgParse")'
```

## Usage

Once you have installed Julia and its ArgParse package, you are ready to train the neural network.

To display the available arguments that can be provided to customize the training, you can execute the following command:

```bash
julia Backpropagation.jl -h
```

As you can see, there are three modes of execution:

* Mode 1: the program reads the data in the `input_file` and divides it in train and test split according to the specified `percentage`.
* Mode 2: the program reads the data in the `input_file`, considering all the data both as the train and test sets.
* Mode 3: the program reads the train data in the `input_file` and the test data in the `input_test_file`.

To further illustrate the usage of the neural network, we show below a sample command:

```bash
julia Backpropagation.jl --input_file data/problem_2.txt --output_name problem_2_example --learning_rate 0.01 --epochs 500 --mode 1 --net_config '[15, 20, 15]' --percentage 0.7
```

Finally, we provide a Python script (`plot_stats.py`) to plot the evolution of the MSE and the Accuracy in the train and test set over the training epochs.