#=
Capa:
- Julia version: 
- Author: guille806
- Date: 2022-02-07
=#

module Capa_pkg

include("Neurona.jl")
using .Neurona_pkg

export Capa

mutable struct Capa
    neuronas::Vector{Neurona}
end

Crear() = Capa(Vector{Neurona}())

function Liberar(capa::Capa)
    for neurona in capa.neuronas
        Neurona_pkg.Liberar(neurona)
    end
    capa = nothing
    GC.gc()
end

function Inicializar(capa::Capa)
    for neurona in capa.neuronas
        Neurona_pkg.Inicializar(neurona, 0.0)
    end
end

function Añadir(capa::Capa, neurona::Neurona)
    push!(capa.neuronas, neurona)
end

function Conectar(capa::Capa, capa_siguiente::Capa, peso_min::Float64, peso_max::Float64)
    for neurona_origen in capa.neuronas
        for neurona_destino in capa_siguiente.neuronas
            # TODO: peso?????
            Neurona_pkg.Conectar(neurona_origen, neurona_destino, peso_min + peso_max)
        end
    end
end

function Conectar(capa::Capa, neurona::Neurona, peso_min::Float64, peso_max::Float64)
    for neurona_origen in capa.neuronas
        # TODO: peso?????
        Neurona_pkg.Conectar(neurona_origen, neurona, peso_min + peso_max)
    end
end

function Disparar(capa::Capa)
    for neurona in capa.neuronas
        Neurona_pkg.Disparar(neurona)
    end
end

function Propagar(capa::Capa)
    for neurona in capa.neuronas
        Neurona_pkg.Propagar(neurona)
    end
end

end

