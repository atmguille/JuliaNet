using Random

"""
    leer2(fichero_de_datos::String) -> (Array{Float64}, Array{Float64})

Lee el fichero indicado en Modo 2, devolviendo (entradas, salidas).
Las entradas contiene un 1 constante al final de cada fila, usado como entrada para el bias de las redes.
# Arguments:
- `fichero_de_datos`: nombre del fichero de datos a leer

"""
function leer2(fichero_de_datos::String)
    file_lines = readlines(fichero_de_datos)
    num_atributos, num_clases = split(file_lines[1])
    num_atributos, num_clases = parse(Int64, num_atributos), parse(Int64, num_clases)

    entradas = Vector{Vector{Float64}}()
    salidas = Vector{Vector{Float64}}()

    for line in file_lines[2:size(file_lines,1)]
        valores = map((x) -> parse(Float64, x), split(line))
        # Añadimos el 1 del bias
        push!(entradas, [valores[1:num_atributos]; [1]])
        # Si las salidas son binarias, convertimos a bipolar
        push!(salidas, map((x) -> x == 0. ? -1. : x, valores[num_atributos+1:size(valores,1)]))
    end

    return entradas, salidas
end

"""
    leer1(fichero_de_datos::String, por::Float64) -> (Array{Float64}, Array{Float64}, Array{Float64}, Array{Float64})

Lee el fichero indicado en Modo 1, devolviendo
(entradas_entrenamiento, salidas_entrenamiento, entradas_test, salidas_test)
# Arguments:
- `fichero_de_datos`: nombre del fichero de datos a leer
- `por`: porcentaje de datos para entrenamiento

"""
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

"""
    leer3(fichero_de_entrenamiento::String, fichero_de_test::String) -> (Array{Float64},
                                                                         Array{Float64},
                                                                         Array{Float64},
                                                                         Array{Float64})

Lee los ficheros indicados en Modo 3, devolviendo
(entradas_entrenamiento, salidas_entrenamiento, entradas_test, salidas_test)
# Arguments:
- `fichero_de_entrenamiento`: nombre del fichero de datos de entrenamiento
- `fichero_de_test`: nombre del fichero de datos de test

"""
function leer3(fichero_de_entrenamiento::String, fichero_de_test::String)
    entradas_entrenamiento, salidas_entrenamiento = leer2(fichero_de_entrenamiento)
    entradas_test, salidas_test = leer2(fichero_de_test)
    return entradas_entrenamiento, salidas_entrenamiento, entradas_test, salidas_test
end

"""
    leer_modo(modo::Int64, parsed_args::Dict) -> (Array{Float64}, Array{Float64}, Array{Float64}, Array{Float64})

Lee los ficheros incluidos en parsed_args en el modo indicado, devolviendo
(entradas_entrenamiento, salidas_entrenamiento, entradas_test, salidas_test)
# Arguments:
- `modo`: modo de lectura
- `parsed_args`: diccionario que debe contener el nombre del fichero a leer en "input_file".
                 Si el modo lo requiere, debe contener el porcentaje en "porcentaje".
                 Si el modo lo requiere, debe contener el nombre del fichero de test en "input_test_file".

"""
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