include("Neurona_pkg.jl")
include("Capa_pkg.jl")
include("RedNeuronal_pkg.jl")
using .Neurona_pkg
using .Capa_pkg
using .RedNeuronal_pkg

using ArgParse
using DelimitedFiles
using Statistics

include("LecturaDatos.jl")
include("utils.jl")


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--input_file"
            help = "Fichero con los valores de entrada para entrenar y probar la red (modos 1 y 2). En el modo 3, únicamente es el fichero de entrenamiento."
            required = true
        "--output_name"
            help = "Nombre base para los ficheros de salida en el que se van a almacenar las predicciones, ecm y tasa de aciertos."
            required = true
        "--tasa_aprendizaje"
            help = "Tasa de aprendizaje del perceptrón multicapa."
            arg_type = Float64
            required = true
        "--epocas"
            help = "Número máximo de épocas para realizar el entrenamiento."
            arg_type = Int64
            required = true
        "--modo"
            help = "Modo de funcionamiento para la lectura de datos."
            arg_type = Int64
            required = true
        "--porcentaje"
            help = "Porcentaje de datos del fichero utilizados en el entrenamiento. Exclusivo del modo 1."
            arg_type = Float64
        "--input_test_file"
            help = "Fichero con los valores de entrada para probar la red. Exclusivo del modo 3."
        "--red_config"
            help = "Lista con la configuración de las capas ocultas de la red. Por ejemplo,
                    '[3,2,1]' indica que la red tendrá 3 capas ocultas, la primera con 3 neuronas,
                    la segunda con 2 neuronas y la tercera con 1."
            arg_type = Vector{Int64}
            required = true
        "--normalizar"
            help = "Si se indica, se normalizan los datos de entrada."
            nargs = 0
    end

    return parse_args(s)
end

"""
    ECM(valores_reales::Vector, prediccion::Vector)

Calcula el error cuadrático medio (ECM) entre los valores reales y la predicción.
# Arguments:
- `valores_reales::Vector`: Valores reales
- `prediccion::Vector`: Predicción

"""
function ECM(valores_reales::Vector{Float64}, prediccion::Vector{Float64})
    return sum(map((x) -> x^2, prediccion-valores_reales)) / size(prediccion, 1)
end

"""
    predicciones_acc_ECM(red::RedNeuronal, entradas::Vector, salidas::Vector) -> (Vector, Float64, Float64, Matrix)

Calcula las predicciones, el accuracy, el error cuadrático medio (ECM) y la matriz de confusión de la red neuronal,
devolviendo (predicciones, accuracy, ECM, matriz_confusion)
# Arguments:
- `red::RedNeuronal`: Red neuronal
- `entradas::Vector`: Valores de entrada de la red
- `salidas::Vector`: Valores esperados de salida de la red

"""
function predicciones_acc_ECM(red::RedNeuronal_pkg.RedNeuronal, entradas::Vector{Vector{Float64}},
                              salidas::Vector{Vector{Float64}})
    predicciones = []
    n_entradas = size(entradas, 1)
    n_clases = size(salidas[1], 1)
    prediccion_clase = repeat([-1.], n_clases)
    ecm = 0
    matriz_confusion = zeros(Int64, (n_clases, n_clases))

    for i in 1:n_entradas
        atributos = entradas[i]
        clases = salidas[i]
        RedNeuronal_pkg.Feedforward(red, atributos)
        prediccion = [neurona.valor_salida for neurona in last(red.capas).neuronas]
        ecm += ECM(clases, prediccion)
        _, index_real = findmax(clases)
        # La clase predicha es la neurona que más se ha activado
        _, index_pred = findmax(prediccion)
        prediccion_clase[index_pred] = 1.
        matriz_confusion[index_pred, index_real] += 1
        push!(predicciones, copy(prediccion_clase))
        prediccion_clase[index_pred] = -1.
    end
    ecm /= n_entradas
    acc = sum(matriz_confusion[i, i] for i in 1:size(matriz_confusion, 1)) / n_entradas

    return predicciones, acc, ecm, matriz_confusion
end


function main()

    parsed_args = parse_commandline()

    modo = parsed_args["modo"]

    ret = leer_modo(modo, parsed_args)

    if ret == nothing
        return
    end

    entradas_entrenamiento, salidas_entrenamiento, entradas_test, salidas_test = ret

    output_name = parsed_args["output_name"]
    tasa_aprendizaje = parsed_args["tasa_aprendizaje"]
    epocas = parsed_args["epocas"]
    red_config = parsed_args["red_config"]
    normalizar = parsed_args["normalizar"]

    # Las entradas ya contienen la constante del bias, luego restamos 1 para el número de atributos
    num_atributos = size(entradas_entrenamiento[1], 1) - 1
    num_clases = size(salidas_entrenamiento[1], 1)

    # Inicializamos la red
    red = RedNeuronal_pkg.CrearRedAleatoria([num_atributos; red_config; num_clases], -0.5, 0.5)
    # Normalizamos los datos de entrada si procede
    if normalizar
        # La función mean y std esperan un tipo distinto al que procesamos. Lo transformamos con reduce
        media_entrenamiento = mean(reduce(hcat, entradas_entrenamiento)', dims=1)
        desv_entrenamiento = std(reduce(hcat, entradas_entrenamiento)', dims=1)
        # Al incluir los datos el sesgo, debemos mantenerlo a 1 pese a normalizar
        media_entrenamiento = [media_entrenamiento[1:num_atributos]; 0.]
        desv_entrenamiento = [desv_entrenamiento[1:num_atributos]; 1.]
        entradas_entrenamiento = map((x) -> (x-media_entrenamiento) ./ desv_entrenamiento, entradas_entrenamiento)
        entradas_test = map((x) -> (x-media_entrenamiento) ./ desv_entrenamiento, entradas_test)
    end

    predicciones_test = []
    array_ecm_train = []
    array_acc_train = []
    array_ecm_test = []
    array_acc_test = []

    for epoch in 1:epocas
        for i in 1:size(entradas_entrenamiento, 1)
            atributos = entradas_entrenamiento[i]

            clases = salidas_entrenamiento[i]
            RedNeuronal_pkg.Feedforward(red, atributos)
            RedNeuronal_pkg.Backpropagation(red, clases, tasa_aprendizaje)
        end

        _, acc_train, ecm_train, mat_conf_train = predicciones_acc_ECM(red, entradas_entrenamiento, salidas_entrenamiento)
        predicciones_test, acc_test, ecm_test, mat_conf_test = predicciones_acc_ECM(red, entradas_test, salidas_test)
        println("Época ", epoch)
        println("ECM Train: ", ecm_train, " ECM Test: ", ecm_test)
        println("Accuracy Train: ", acc_train, " Accuracy Test: ", acc_test)
        println("Matriz Confusión Train: ", mat_conf_train, " Matriz Confusión Test: ", mat_conf_test)
        push!(array_ecm_train, ecm_train)
        push!(array_ecm_test, ecm_test)
        push!(array_acc_train, acc_train)
        push!(array_acc_test, acc_test)
    end

    writedlm(output_name * "_pred_test.txt", predicciones_test)
    writedlm(output_name * "_ecm_train.txt", array_ecm_train)
    writedlm(output_name * "_ecm_test.txt", array_ecm_test)
    writedlm(output_name * "_acc_train.txt", array_acc_train)
    writedlm(output_name * "_acc_test.txt", array_acc_test)

    RedNeuronal_pkg.Liberar(red)

end

main()