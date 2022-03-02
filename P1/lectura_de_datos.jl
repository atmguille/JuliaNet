using Random

function leer2(fichero_de_datos::String)
    file_lines = readlines(fichero_de_datos)
    num_atributos, num_clases = split(file_lines[1])
    num_atributos, num_clases = parse(Int64, num_atributos), parse(Int64, num_clases)

    entradas = Vector{Vector{Float64}}()
    salidas = Vector{Vector{Float64}}()

    for line in file_lines[2:size(file_lines,1)]
        valores = map((x) -> parse(Float64, x), split(line))
        push!(entradas, [valores[1:num_atributos]; [1]])  # Añadimos el 1 del bias
        push!(salidas, valores[num_atributos+1:size(valores,1)])
    end

    return entradas, salidas
end

function leer1(fichero_de_datos::String, por::Float64)
    entradas, salidas = leer2(fichero_de_datos)

    seed = round(Int64, time() * 1000)
    shuffle!(Random.seed!(seed), entradas)
    shuffle!(Random.seed!(seed), salidas)

    entradas_entrenamiento = entradas[1:round(Int64, size(entradas,1)*por)]
    entradas_test = entradas[round(Int64, size(entradas,1)*por)+1:size(entradas,1)]
    salidas_entrenamiento = salidas[1:round(Int64, size(salidas,1)*por)]
    salidas_test = salidas[round(Int64, size(salidas,1)*por)+1:size(salidas,1)]
    return entradas_entrenamiento, salidas_entrenamiento, entradas_test, salidas_test
end

function leer3(fichero_de_entrenamiento::String, fichero_de_test::String)
    entradas_entrenamiento, salidas_entrenamiento = leer2(fichero_de_entrenamiento)
    entradas_test, salidas_test = leer2(fichero_de_test)
    return entradas_entrenamiento, salidas_entrenamiento, entradas_test, salidas_test
end


function leer_modo(modo::Int64, parsed_args::Dict)
    input_file = parsed_args["input_file"]
    
    if modo == 1
        porcentaje = parsed_args["porcentaje"]
        if porcentaje == nothing
            println("Es necesario indicar el porcentaje en el modo 1.")
            return nothing
        end
        entradas_entrenamiento, salidas_entrenamiento, entradas_test, salidas_test = leer1(input_file, porcentaje)
    elseif modo == 2
        entradas_entrenamiento, salidas_entrenamiento = leer2(input_file)
        entradas_test, salidas_test = entradas_entrenamiento, salidas_entrenamiento
    elseif modo == 3
        input_test_file = parsed_args["input_test_file"]
        if input_test_file == nothing
            println("Es necesario indicar el fichero utilizado para test en el modo 3.")
            return nothing
        end
        entradas_entrenamiento, salidas_entrenamiento, entradas_test, salidas_test = leer3(input_file, input_test_file)
    end

    return entradas_entrenamiento, salidas_entrenamiento, entradas_test, salidas_test
end