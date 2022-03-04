module RedNeuronal_pkg

using ..Capa_pkg

mutable struct RedNeuronal
    capas::Vector{Capa}
end

"""
    Crear() -> RedNeuronal

Crea una red neuronal vacía

"""
Crear() = RedNeuronal(Vector{Capa}())

"""
    Liberar(red_neuronal::RedNeuronal)

Libera la red neuronal y todas las capas que contiene
# Arguments:
- `red_neuronal::RedNeuronal`: Red neuronal a liberar

"""
function Liberar(red_neuronal::RedNeuronal)
    for capa in red_neuronal.capas
        Capa_pkg.Liberar(capa)
    end
    red_neuronal = nothing
    GC.gc()
end

"""
    Inicializar(red_neuronal::RedNeuronal)

Inicializa todas las neuronas de la red neuronal a 0
# Arguments:
- `red_neuronal::RedNeuronal`: Red neuronal a inicializar

"""
function Inicializar(red_neuronal::RedNeuronal)
    for capa in red_neuronal.capas
        Capa_pkg.Inicializar(capa)
    end
end

"""
    Añadir(red_neuronal::RedNeuronal, capa::Capa)

Añade una capa a la red neuronal
# Arguments:
- `red_neuronal::RedNeuronal`: Red neuronal a la que añadir la capa
- `capa::Capa`: Capa a añadir

"""
function Añadir(red_neuronal::RedNeuronal, capa::Capa)
    push!(red_neuronal.capas, capa)
end

"""
    Disparar(red_neuronal::RedNeuronal)

Dispara todas las capas de la red neuronal
# Arguments:
- `red_neuronal::RedNeuronal`: Red neuronal a disparar

"""
function Disparar(red_neuronal::RedNeuronal)
    for capa in red_neuronal.capas
        Capa_pkg.Disparar(capa)
    end
end

"""
    Propagar(red_neuronal::RedNeuronal)

Propaga todas las capas de la red neuronal
# Arguments:
- `red_neuronal::RedNeuronal`: Red neuronal a propagar

"""
function Propagar(red_neuronal::RedNeuronal)
    for capa in red_neuronal.capas
        Capa_pkg.Propagar(capa)
    end
end

end