"""
    avanzar_ciclo(red::RedNeuronal, valores_entrada::Vector)

Avanza un ciclo de la red neuronal. Consiste en:
* Inicializar la capa de entrada a los valores de entrada
* Disparar todas las neuronas de la red
* Descargar las neuronas de la red (inicialización de valores de entrada a 0)
* Propagar todas las neuronas de la red

# Arguments:
- `red::RedNeuronal`: Red neuronal
- `valores_entrada::Vector`: Valores de entrada

"""
function avanzar_ciclo(red::RedNeuronal_pkg.RedNeuronal, valores_entrada::Vector{Float64})
    capa_entrada = red.capas[1]
    for i in 1:size(valores_entrada, 1)
        Neurona_pkg.Inicializar(capa_entrada.neuronas[i], valores_entrada[i])
    end
    RedNeuronal_pkg.Disparar(red)
    RedNeuronal_pkg.Inicializar(red)
    RedNeuronal_pkg.Propagar(red)
end

"""
    print_pesos(red::RedNeuronal)

Imprime todos los pesos de la red neuronal.
# Arguments:
- `red::RedNeuronal`: Red neuronal

"""
function print_pesos(red::RedNeuronal_pkg.RedNeuronal)
    for capa in red.capas
        for neurona in capa.neuronas
            for conexion in neurona.conexiones
                println(conexion.peso)
            end
        end
    end
    println("----------------")
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
    predicciones_acc_ECM(red::RedNeuronal, entradas::Vector, salidas::Vector) -> (Vector, Float64, Float64)

Calcula las predicciones, el accuracy y el error cuadrático medio (ECM) de la red neuronal,
devolviendo (predicciones, accuracy, ECM)
# Arguments:
- `red::RedNeuronal`: Red neuronal
- `entradas::Vector`: Valores de entrada de la red
- `salidas::Vector`: Valores esperados de salida de la red

"""
function predicciones_acc_ECM(red::RedNeuronal_pkg.RedNeuronal, entradas::Vector{Vector{Float64}},
                            salidas::Vector{Vector{Float64}})
    predicciones = []
    
    ecm = 0
    acc = 0
    for i in 1:size(entradas,1)
        atributos = entradas[i]
        clases = salidas[i]
        avanzar_ciclo(red, atributos)
        Capa_pkg.Disparar(last(red.capas))
        prediccion = [neurona.valor_salida for neurona in last(red.capas).neuronas]
        ecm += ECM(clases, prediccion)
        acc += (prediccion == clases ? 1 : 0) 
        push!(predicciones, prediccion)
    end
    ecm /= size(entradas,1)
    acc /= size(salidas, 1)
    
    return predicciones, acc, ecm 
end

"""
    main_generico(red::RedNeuronal, entradas_entrenamiento::Vector, salidas_entrenamiento::Vector,
                  entradas_test::Vector, salidas_test::Vector, funcion_entrenamiento::Function,
                  parsed_args::Dict)

Función genérica para entrenar una red neuronal para un conjunto de datos de entrenamiento
y obtener las predicciones para un conjunto de datos de test.
# Arguments:
- `red::RedNeuronal`: Red neuronal
- `entradas_entrenamiento::Vector`: Valores de entrada de entrenamiento de la red
- `salidas_entrenamiento::Vector`: Valores esperados de salida de entrenamiento de la red
- `entradas_test::Vector`: Valores de entrada de test de la red
- `salidas_test::Vector`: Valores esperados de salida de test de la red
- `funcion_entrenamiento::Function`: Función usada para el entrenamiento de la red
- `parsed_args::Dict`: Argumentos de entrada al programa

"""
function main_generico(red::RedNeuronal_pkg.RedNeuronal, entradas_entrenamiento::Vector{Vector{Float64}},
                       salidas_entrenamiento::Vector{Vector{Float64}}, entradas_test::Vector{Vector{Float64}},
                       salidas_test::Vector{Vector{Float64}}, funcion_entrenamiento::Function, parsed_args::Dict)

    output_file = parsed_args["output_file"]
    tasa_aprendizaje = parsed_args["tasa_aprendizaje"]
    max_epocas = parsed_args["max_epocas"]
    tolerancia = get(parsed_args, "tolerancia", 0.0)

    # Las entradas ya contienen la constante del bias, luego restamos 1 para el número de atributos
    num_atributos = size(entradas_entrenamiento[1], 1) - 1
    num_clases = size(salidas_entrenamiento[1], 1)

    predicciones_test = []

    fin_entrenamiento = true
    for epoch in 1:max_epocas
        # Reset flag
        fin_entrenamiento = true
        for i in 1:size(entradas_entrenamiento, 1)
            atributos = entradas_entrenamiento[i]
            clases = salidas_entrenamiento[i]
            avanzar_ciclo(red, atributos)
            fin_entrenamiento = fin_entrenamiento & funcion_entrenamiento(red, tasa_aprendizaje,
                                                                          num_atributos+1, atributos,
                                                                          num_clases, clases, tolerancia)
        end

        _, acc_train, ecm_train = predicciones_acc_ECM(red, entradas_entrenamiento, salidas_entrenamiento)
        predicciones_test, acc_test, ecm_test = predicciones_acc_ECM(red, entradas_test, salidas_test)
        println("Época ", epoch)
        println("ECM Train: ", ecm_train, " ECM Test: ", ecm_test)
        println("Accuracy Train: ", acc_train, " Accuracy Test: ", acc_test)

        if fin_entrenamiento
            println("Entrenamiento finalizado por convergencia en los pesos.")
            break
        end

    end

    if !fin_entrenamiento
        println("Entrenamiento finalizado: número máximo de épocas alcanzado.")
    end

    writedlm(output_file, predicciones_test)

    RedNeuronal_pkg.Liberar(red)

end