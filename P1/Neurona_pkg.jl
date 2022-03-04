module Conexion_pkg

export Conexion

abstract type AbstractNeurona end

mutable struct Conexion{T<:AbstractNeurona}
    peso::Float64
    peso_anterior::Float64
    valor::Float64
    neurona::T
end

"""
    Crear(peso::Float64, neurona::Neurona) -> Conexion

# Arguments:

- `peso::Float64`: Peso de la conexion
- `neurona::Neurona`: Neurona a la que se conecta

"""
Crear(peso::Float64, neurona::AbstractNeurona) = Conexion(peso, 0.0, 0.0, neurona)


"""
    Liberar(conexion::Conexion)

# Arguments:

- `conexion::Conexion`: Conexion a liberar
"""
function Liberar(conexion::Conexion)
    conexion = nothing
    GC.gc()
end

"""
    Propagar(conexion::Conexion, valor::Float64)

# Arguments:

- `conexion::Conexion`: Conexion por la que propagar el valor
- `valor::Float64`: Valor a propagar entre neuronas a tavés de la conexión
"""
function Propagar(conexion::Conexion, valor::Float64)
    conexion.neurona.valor_entrada +=  valor * conexion.peso
    conexion.valor = valor
end

end

module Neurona_pkg

using ..Conexion_pkg

export Neurona

@enum Tipo begin
    Directa
    Sesgo
    McCulloch
    Perceptron
    Adaline
end


mutable struct Neurona <: Conexion_pkg.AbstractNeurona
    tipo::Tipo
    umbral::Float64
    valor_entrada::Float64
    valor_salida::Float64
    conexiones::Vector{Conexion}
end

"""
    Crear(umbral::Float64, tipo::Tipo) -> Neurona

# Arguments:

- `umbral::Float64`: Umbral de activación de la neurona
- `tipo::Tipo`: Tipo de la neurona (Directa, Sesgo, McCulloch, Perceptron ó Adaline)

"""
Crear(umbral::Float64, tipo::Tipo) = Neurona(tipo, umbral, 0.0, 0.0, Vector{Conexion}())

"""
    Liberar(neurona::Neurona)

Libera la neurona así como las conexiones que salen de ella
# Arguments

- `neurona::Neurona`: Neurona a liberar

"""
function Liberar(neurona::Neurona)
    for conexion in neurona.conexiones
        Conexion_pkg.Liberar(conexion)
    end
    neurona = nothing
    GC.gc()
end

"""
    Inicializar(neurona::Neurona, x::Float64)

# Arguments:
- `neurona::Neurona`: Neurona a inicializar
- `x::Float64`: Valor de entrada de la neurona

"""
Inicializar(neurona::Neurona, x::Float64) = neurona.valor_entrada = x

"""
    Conectar(neurona_origen::Neurona, neurona_destino::Neurona, peso::Float64)

Se crea una conexión entre las neuronas indicadas, añadiéndola a la lista de conexiones de la neurona origen
# Arguments:
- `neurona_origen::Neurona`: Neurona origen de la conexión
- `neurona_destino::Neurona`: Neurona destino de la conexión
- `peso::Float64`: Peso de la conexión

"""
Conectar(neurona_origen::Neurona, neurona_destino::Neurona, peso::Float64) = push!(neurona_origen.conexiones,
                                                                                   Conexion_pkg.Crear(peso, neurona_destino))

"""
    Disparar(neurona::Neurona)

Procesa el valor de entrada de la neurona con la activación correspondiente según su tipo
# Arguments:
- `neurona::Neurona`: Neurona a disparar

"""
function Disparar(neurona::Neurona)
    if neurona.tipo == Directa
        neurona.valor_salida = neurona.valor_entrada
    elseif neurona.tipo == Sesgo
        neurona.valor_salida = 1.0
    elseif neurona.tipo == McCulloch
        neurona.valor_salida = neurona.valor_entrada >= neurona.umbral ? 1.0 : 0.0
    elseif neurona.tipo == Perceptron
        if neurona.valor_entrada > neurona.umbral
            neurona.valor_salida = 1.0
        elseif neurona.valor_entrada < -1*neurona.umbral
            neurona.valor_salida = -1.0
        else
            neurona.valor_salida = 0.0
        end
    elseif neurona.tipo == Adaline
        if neurona.valor_entrada >= 0
            neurona.valor_salida = 1.0
        else
            neurona.valor_salida = -1.0
        end
    end
end

"""
    Propagar(neurona::Neurona)

Propaga el valor de salida de la neurona a través de sus conexiones
# Arguments:
- `neurona::Neurona`: Neurona a propagar

"""
function Propagar(neurona::Neurona)
    for conexion in neurona.conexiones
        Conexion_pkg.Propagar(conexion, neurona.valor_salida)
    end
end

end
