include("Neurona_pkg.jl")
include("Capa_pkg.jl")
include("RedNeuronal_pkg.jl")
using .Neurona_pkg
using .Capa_pkg
using .RedNeuronal_pkg

include("LecturaDatos.jl")
include("utils.jl")

using ArgParse  # import Pkg; Pkg.add("ArgParse")
using DelimitedFiles

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--input_file"
            help = "Fichero con los valores de entrada para entrenar y probar la red (modos 1 y 2). En el modo 3, únicamente es el fichero de entrenamiento."
            required = true
        "--output_file"
            help = "Fichero en el que se van a almacenar los valores de la neurona en cada instante de tiempo."
            required = true
        "--umbral"
            help = "Umbral a utilizar para entrenar el perceptrón."
            arg_type = Float64
            required = true
        "--tasa_aprendizaje"
            help = "Tasa de aprendizaje del perceptrón."
            arg_type = Float64
            required = true
        "--max_epocas"
            help = "Número máximo de épocas para realizar el entrenamiento."
            arg_type = Int64
            required = true
        "--modo"
            help = "Modo de funcionamiento para la lectura de datos."
            arg_type = Int64
            required = true
        "--porcentaje"
            help = "Porcentaje de datos del fichero utilizados en el entrenamiento. Exclusivo del modo 2."
            arg_type = Float64
        "--input_test_file"
            help = "Fichero con los valores de entrada para probar la red. Exclusivo del modo 3."
    end

    return parse_args(s)
end


function crear_perceptron(num_atributos::Int64, num_clases::Int64, umbral::Float64)
    red = RedNeuronal_pkg.Crear()

    capa_entrada = Capa_pkg.Crear()

    for i in 1:num_atributos
        x = Neurona_pkg.Crear(1.0, Neurona_pkg.Directa)
        Capa_pkg.Añadir(capa_entrada, x)
    end

    sesgo = Neurona_pkg.Crear(1.0, Neurona_pkg.Sesgo)
    Capa_pkg.Añadir(capa_entrada, sesgo)
    
    RedNeuronal_pkg.Añadir(red, capa_entrada)

    capa_salida = Capa_pkg.Crear()

    for i in 1:num_clases
        x = Neurona_pkg.Crear(umbral, Neurona_pkg.Perceptron)
        Capa_pkg.Añadir(capa_salida, x)
    end

    RedNeuronal_pkg.Añadir(red, capa_salida)
    
    for neur_in in capa_entrada.neuronas
        for neur_out in capa_salida.neuronas
            Neurona_pkg.Conectar(neur_in, neur_out, 0.0)
        end
    end

    return red
end

# TODO: tolerancia vacia
function entrenamiento_perceptron(red::RedNeuronal_pkg.RedNeuronal, tasa_aprendizaje::Float64,
    num_atributos::Int64, atributos::Vector{Float64}, num_clases::Int64, clases_verdaderas::Vector{Float64}, tolerancia)
    capa_entrada = red.capas[1]
    capa_salida = red.capas[2]
    # Los valores de salida de la última capa deben ser actualizados para obtener la respuesta final
    Capa_pkg.Disparar(capa_salida)

    fin_entrenamiento = true

    for clase_index in 1:num_clases
        if clases_verdaderas[clase_index] != capa_salida.neuronas[clase_index].valor_salida
            fin_entrenamiento = false
            for atributo_index in 1:num_atributos
                conexion = capa_entrada.neuronas[atributo_index].conexiones[clase_index]
                conexion.peso_anterior = conexion.peso
                conexion.peso += tasa_aprendizaje * clases_verdaderas[clase_index] * atributos[atributo_index]
            end
        end
    end
    
    return fin_entrenamiento
end


function ECM(prediccion::Vector{Float64}, capa_salida::Capa_pkg.Capa)
    valores_reales = [neurona.valor_salida for neurona in capa_salida.neuronas]
    return sum(map((x) -> x^2, prediccion-valores_reales)) / size(prediccion, 1)
end

function ECM(red::RedNeuronal_pkg.RedNeuronal, entradas::Vector{Vector{Float64}}, salidas::Vector{Vector{Float64}})
    ecm = 0
    for i in 1:size(entradas,1)
        atributos = entradas[i]
        clases = salidas[i]
        avanzar_ciclo(red, atributos)
        Capa_pkg.Disparar(last(red.capas))
        ecm += ECM(clases, last(red.capas))
    end
    ecm /= size(entradas,1)
    return ecm
end

function main()

    parsed_args = parse_commandline()

    umbral = parsed_args["umbral"]
    modo = parsed_args["modo"]

    ret = leer_modo(modo, parsed_args)

    if ret == nothing
        return
    end
    
    entradas_entrenamiento, salidas_entrenamiento, entradas_test, salidas_test = ret

    num_atributos = size(entradas_entrenamiento[1], 1) - 1
    num_clases = size(salidas_entrenamiento[1], 1)
    perceptron = crear_perceptron(num_atributos, num_clases, umbral)

    main_generico(perceptron, entradas_entrenamiento, salidas_entrenamiento, entradas_test, salidas_test, entrenamiento_perceptron, parsed_args)

end

main()