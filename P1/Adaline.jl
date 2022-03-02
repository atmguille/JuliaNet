include("Neurona_pkg.jl")
include("Capa_pkg.jl")
include("RedNeuronal_pkg.jl")
using .Neurona_pkg
using .Capa_pkg
using .RedNeuronal_pkg

include("lectura_de_datos.jl")
include("utils.jl")

using ArgParse  # import Pkg; Pkg.add("ArgParse")
using DelimitedFiles

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--input_file"
            help = "Fichero con los valores de entrada para la red frio-calor."
            required = true
        "--output_file"
            help = "Fichero en el que se van a almacenar los valores de la neurona en cada instante de tiempo."
            required = true
        "--tasa_aprendizaje"
            help = "Tasa de aprendizaje del adaline."
            arg_type = Float64
            required = true
        "--max_epocas"
            help = "Número máximo de épocas para realizar el entrenamiento."
            arg_type = Int64
            required = true
        "--tolerancia"
            help = "Si el mayor cambio de pesos es menor que esta tolerancia, el entrenamiento finaliza."
            arg_type = Float64
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


function crear_adaline(num_atributos::Int64, num_clases::Int64)
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
        x = Neurona_pkg.Crear(0.0, Neurona_pkg.Adaline)
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


function entrenamiento_adaline(red::RedNeuronal_pkg.RedNeuronal, tasa_aprendizaje::Float64,
    num_atributos::Int64, atributos::Vector{Float64}, num_clases::Int64,
    clases_verdaderas::Vector{Float64}, tolerancia::Float64)
    capa_entrada = red.capas[1]
    capa_salida = red.capas[2]

    max_delta = -Inf

    for clase_index in 1:num_clases
        for atributo_index in 1:num_atributos
            conexion = capa_entrada.neuronas[atributo_index].conexiones[clase_index]
            conexion.peso_anterior = conexion.peso
            # En fase de entrenamiento, se usa el valor de entrada como respuesta (sin activación) # TODO: ojo que al predecir sí que hay que usar el valor de salida
            y_in = capa_salida.neuronas[clase_index].valor_entrada
            t = clases_verdaderas[clase_index]
            delta = tasa_aprendizaje * (t-y_in) * atributos[atributo_index]
            conexion.peso += delta

            if delta > max_delta
                max_delta = delta
            end
        end
    end

    if max_delta < tolerancia
        fin_entrenamiento = true
    else
        fin_entrenamiento = false
    end

    return fin_entrenamiento
end


function main()

    parsed_args = parse_commandline()

    modo = parsed_args["modo"]

    ret = leer_modo(modo, parsed_args)

    if ret == nothing
        return
    end
    
    entradas_entrenamiento, salidas_entrenamiento, entradas_test, salidas_test = ret

    num_atributos = size(entradas_entrenamiento[1], 1) - 1
    num_clases = size(salidas_entrenamiento[1], 1)
    adaline = crear_adaline(num_atributos, num_clases)

    main_generico(adaline, entradas_entrenamiento, salidas_entrenamiento, entradas_test, salidas_test, entrenamiento_adaline, parsed_args)

end

main()