# JuliaNet

This repository contains a educational implementation of a generic deep neural network in Julia. Its purpose is to facilitate the understanding of how deep neural networks work, both for the feed-forward and the back-propagation. It comes from a more specific version initially developed for the Neurocomputation course at UAM, which imposed the structure of the library for the neural network (check it [here](https://github.com/AlejandroSantorum/Apuntes_Mat_IngInf/blob/master/5_Curso/Neurocomputaci%C3%B3n/Pr%C3%A1cticas/Pr%C3%A1ctica1/Enunciado_Pr%C3%A1ctica1.pdf)). Due to this structural imposition for educational purposes, we could not make the library as efficient as possible (via matrix multiplications instead of for loops). However, it is still very efficient thanks to Julia, since it is **10 times faster** than the equivalent Python implementation that the rest of the students did.

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

