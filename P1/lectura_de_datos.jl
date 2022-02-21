using Random

function leer2(fichero_de_datos::String)
    file_lines = readlines(fichero_de_datos)
    num_atributos, num_clases = split(file_lines[1], " ")
    num_atributos, num_clases = parse(Int64, num_atributos), parse(Int64, num_clases)

    entradas = Vector{Vector{Float64}}()
    salidas = Vector{Vector{Float64}}()

    for line in file_lines[2:size(file_lines,1)]
        valores = map((x) -> parse(Float64, x), split(line, "  "))  # TODO: fix double space
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