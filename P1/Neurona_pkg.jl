#=
Neurona:
- Julia version: 
- Author: guille806
- Date: 2022-02-07
=#

module Conexion_pkg

export Conexion

abstract type AbstractNeurona end

mutable struct Conexion{T<:AbstractNeurona}
    peso::Float64
    peso_anterior::Float64
    valor::Float64
    neurona::T
end

Crear(peso::Float64, neurona::AbstractNeurona) = Conexion(peso, 0.0, 0.0, neurona)

function Liberar(conexion::Conexion)
    conexion = nothing
    GC.gc()
end

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
end


mutable struct Neurona <: Conexion_pkg.AbstractNeurona
    tipo::Tipo
    umbral::Float64
    valor_entrada::Float64
    valor_salida::Float64
    conexiones::Vector{Conexion}
end

"""
TODO: doc
"""
Crear(umbral::Float64, tipo::Tipo) = Neurona(tipo, umbral, 0.0, 0.0, Vector{Conexion}())

"""
    Liberar(neurona::Neurona)

# Arguments

- `neurona::Neurona`:


"""
function Liberar(neurona::Neurona)
    for conexion in neurona.conexiones
        Conexion_pkg.Liberar(conexion)
    end
    neurona = nothing
    GC.gc()
end

Inicializar(neurona::Neurona, x::Float64) = neurona.valor_entrada = x

Conectar(neurona_origen::Neurona, neurona_destino::Neurona, peso::Float64) = push!(neurona_origen.conexiones, Conexion_pkg.Crear(peso, neurona_destino))

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
    end
end

function Propagar(neurona::Neurona)
    for conexion in neurona.conexiones
        Conexion_pkg.Propagar(conexion, neurona.valor_salida)
    end
end

end
