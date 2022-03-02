function avanzar_ciclo(red::RedNeuronal_pkg.RedNeuronal, valores_entrada::Vector{Float64})
    capa_entrada = red.capas[1]
    for i in 1:size(valores_entrada, 1)
        Neurona_pkg.Inicializar(capa_entrada.neuronas[i], valores_entrada[i])
    end
    RedNeuronal_pkg.Disparar(red)
    RedNeuronal_pkg.Inicializar(red)
    RedNeuronal_pkg.Propagar(red)
end


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


function ECM(valores_reales::Vector{Float64}, prediccion::Vector{Float64})
    return sum(map((x) -> x^2, prediccion-valores_reales)) / size(prediccion, 1)
end

function predicciones_y_ECM(red::RedNeuronal_pkg.RedNeuronal, entradas::Vector{Vector{Float64}}, salidas::Vector{Vector{Float64}})
    predicciones = []
    
    ecm = 0
    for i in 1:size(entradas,1)
        atributos = entradas[i]
        clases = salidas[i]
        avanzar_ciclo(red, atributos)
        Capa_pkg.Disparar(last(red.capas))
        prediccion = [neurona.valor_salida for neurona in last(red.capas).neuronas]
        ecm += ECM(clases, prediccion)
        push!(predicciones, prediccion)
    end
    ecm /= size(entradas,1)
    
    return ecm, predicciones 
end

function main_generico(red::RedNeuronal_pkg.RedNeuronal, entradas_entrenamiento::Vector{Vector{Float64}}, salidas_entrenamiento::Vector{Vector{Float64}}, 
    entradas_test::Vector{Vector{Float64}}, salidas_test::Vector{Vector{Float64}}, funcion_entrenamiento::Function, parsed_args::Dict)

    output_file = parsed_args["output_file"]
    tasa_aprendizaje = parsed_args["tasa_aprendizaje"]
    max_epocas = parsed_args["max_epocas"]
    tolerancia = get(parsed_args, "tolerancia", 0.0)

    num_atributos = size(entradas_entrenamiento[1], 1) - 1 # TODO: comentar bias en atributos
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
            fin_entrenamiento = fin_entrenamiento & funcion_entrenamiento(red, tasa_aprendizaje, num_atributos+1, atributos, num_clases, clases, tolerancia)
        end

        ecm_train, _ = predicciones_y_ECM(red, entradas_entrenamiento, salidas_entrenamiento)
        ecm_test, predicciones_test = predicciones_y_ECM(red, entradas_test, salidas_test)
        println("Época ", epoch)
        println("ECM Train: ", ecm_train, " ECM Test: ", ecm_test)


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