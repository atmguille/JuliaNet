module Capa_pkg

using ..Neurona_pkg

export Capa

mutable struct Capa
    neuronas::Vector{Neurona}
end

"""
    Crear() -> Capa

Crea una capa vacía.

"""
Crear() = Capa(Vector{Neurona}())

"""
    Liberar(capa::Capa)

Libera la capa y las neuronas que contiene
# Arguments:
- `capa::Capa`: Capa a liberar

"""
function Liberar(capa::Capa)
    for neurona in capa.neuronas
        Neurona_pkg.Liberar(neurona)
    end
    capa = nothing
    GC.gc()
end

"""
    Inicializar(capa::Capa)

Inicializa las neuronas de la capa a valor de entrada 0
# Arguments:
- `capa::Capa`: Capa a inicializar

"""
function Inicializar(capa::Capa)
    for neurona in capa.neuronas
        Neurona_pkg.Inicializar(neurona, 0.0)
    end
end

"""
    Añadir(capa::Capa, neurona::Neurona)

Añade una neurona a la capa
# Arguments:
- `capa::Capa`: Capa a la que añadir la neurona
- `neurona::Neurona`: Neurona a añadir

"""
function Añadir(capa::Capa, neurona::Neurona)
    push!(capa.neuronas, neurona)
end

"""
    Conectar(capa::Capa, neurona::Neurona, peso_min::Float64, peso_max::Float64)

Conecta todas las neuronas de la capa indicada, con la neurona indicada, con pesos aleatorios entre peso_min y peso_max
# Arguments:
- `capa::Capa`: Capa de salida
- `neurona::Neurona`: Neurona de entrada
- `peso_min::Float64`: Peso mínimo
- `peso_max::Float64`: Peso máximo

"""
function Conectar(capa::Capa, neurona::Neurona, peso_min::Float64, peso_max::Float64)
    for neurona_origen in capa.neuronas
        peso = rand() * (peso_max - peso_min) + peso_min
        Neurona_pkg.Conectar(neurona_origen, neurona, peso)
    end
end

"""
    Conectar(capa::Capa, capa_siguiente::Capa, peso_min::Float64, peso_max::Float64)

Conecta todas las neuronas entre las capas indicadas, con pesos aleatorios entre peso_min y peso_max
# Arguments:
- `capa::Capa`: Capa de salida
- `capa_siguiente::Capa`: Capa de entrada
- `peso_min::Float64`: Peso mínimo
- `peso_max::Float64`: Peso máximo

"""
function Conectar(capa::Capa, capa_siguiente::Capa, peso_min::Float64, peso_max::Float64)
    for neurona_destino in capa_siguiente.neuronas
        Conectar(capa, neurona_destino, peso_min, peso_max)
    end
end

"""
    Disparar(capa::Capa)

Dispara todas las neuronas de la capa
# Arguments:
- `capa::Capa`: Capa a disparar

"""
function Disparar(capa::Capa)
    for neurona in capa.neuronas
        Neurona_pkg.Disparar(neurona)
    end
end

"""
    Propagar(capa::Capa)

Propaga todas las neuronas de la capa
# Arguments:
- `capa::Capa`: Capa a propagar

"""
function Propagar(capa::Capa)
    for neurona in capa.neuronas
        Neurona_pkg.Propagar(neurona)
    end
end

end

