include("Neurona_pkg.jl")
include("Capa_pkg.jl")
include("RedNeuronal_pkg.jl")
using .Neurona_pkg
using .Capa_pkg
using .RedNeuronal_pkg

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
    end

    return parse_args(s)
end


function crear_red_frio_calor()
    red = RedNeuronal_pkg.Crear()

    x1 = Neurona_pkg.Crear(1.0, Neurona_pkg.Directa)
    x2 = Neurona_pkg.Crear(1.0, Neurona_pkg.Directa)
    capa_entrada = Capa_pkg.Crear()
    Capa_pkg.Añadir(capa_entrada, x1)
    Capa_pkg.Añadir(capa_entrada, x2)
    RedNeuronal_pkg.Añadir(red, capa_entrada)

    z1 = Neurona_pkg.Crear(2.0, Neurona_pkg.McCulloch)
    z2 = Neurona_pkg.Crear(2.0, Neurona_pkg.McCulloch)
    capa_oculta = Capa_pkg.Crear()
    Capa_pkg.Añadir(capa_oculta, z1)
    Capa_pkg.Añadir(capa_oculta, z2)
    RedNeuronal_pkg.Añadir(red, capa_oculta)

    y1 = Neurona_pkg.Crear(2.0, Neurona_pkg.McCulloch)
    y2 = Neurona_pkg.Crear(2.0, Neurona_pkg.McCulloch)
    capa_salida = Capa_pkg.Crear()
    Capa_pkg.Añadir(capa_salida, y1)
    Capa_pkg.Añadir(capa_salida, y2)
    RedNeuronal_pkg.Añadir(red, capa_salida)

    Neurona_pkg.Conectar(x1, y1, 2.0)
    #
    Neurona_pkg.Conectar(x2, z1, -1.0)
    Neurona_pkg.Conectar(x2, z2, 2.0)
    Neurona_pkg.Conectar(x2, y2, 1.0)
    #
    Neurona_pkg.Conectar(z1, y1, 2.0)
    #
    Neurona_pkg.Conectar(z2, z1, 2.0)
    Neurona_pkg.Conectar(z2, y2, 1.0)

    return red, x1, x2
end

function avanzar_ciclo(red::RedNeuronal_pkg.RedNeuronal, x1::Neurona_pkg.Neurona, calor::Float64, x2::Neurona_pkg.Neurona, frio::Float64)
    Neurona_pkg.Inicializar(x1, calor)
    Neurona_pkg.Inicializar(x2, frio)
    RedNeuronal_pkg.Disparar(red)
    RedNeuronal_pkg.Inicializar(red)
    RedNeuronal_pkg.Propagar(red)
end


function main()

    parsed_args = parse_commandline()

    input_file = parsed_args["input_file"]
    output_file = parsed_args["output_file"]

    """
    if size(ARGS) != (2,)
        println("Número incorrecto de argumentos. Debe ejecutar:")
        println("julia FrioCalor.jl [input_file] [output_file]")
        return 
    end

    input_file = ARGS[1]
    output_file = ARGS[2]
    """

    red, x1, x2 = crear_red_frio_calor()

    valores = Vector{Vector{String}}()
    push!(valores, ["x1", "x2", "z1", "z2", "y1", "y2"])

    for line in readlines(input_file)
        calor, frio = split(line, " ")
        avanzar_ciclo(red, x1, parse(Float64, calor), x2, parse(Float64, frio))
        push!(valores, [string(convert(Int64, neurona.valor_salida)) for capa in red.capas for neurona in capa.neuronas])
    end

    # TODO: Comentar
    avanzar_ciclo(red, x1, 0.0, x2, 0.0)
    push!(valores, [string(convert(Int64, neurona.valor_salida)) for capa in red.capas for neurona in capa.neuronas])

    avanzar_ciclo(red, x1, 0.0, x2, 0.0)
    push!(valores, [string(convert(Int64, neurona.valor_salida)) for capa in red.capas for neurona in capa.neuronas])
    
    writedlm(output_file, valores)

    RedNeuronal_pkg.Liberar(red)

end

main()