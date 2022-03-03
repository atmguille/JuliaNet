#=
RedNeuronal:
- Julia version: 
- Author: guille806
- Date: 2022-02-07
=#

module RedNeuronal_pkg

using ..Capa_pkg

mutable struct RedNeuronal
    capas::Vector{Capa}
end

Crear() = RedNeuronal(Vector{Capa}())

function Liberar(red_neuronal::RedNeuronal)
    for capa in red_neuronal.capas
        Capa_pkg.Liberar(capa)
    end
    red_neuronal = nothing
    GC.gc()
end

function Inicializar(red_neuronal::RedNeuronal)
    for capa in red_neuronal.capas
        Capa_pkg.Inicializar(capa)
    end
end

function Añadir(red_neuronal::RedNeuronal, capa::Capa)
    push!(red_neuronal.capas, capa)
end

function Disparar(red_neuronal::RedNeuronal)
    for capa in red_neuronal.capas
        Capa_pkg.Disparar(capa)
    end
end

function Propagar(red_neuronal::RedNeuronal)
    for capa in red_neuronal.capas
        Capa_pkg.Propagar(capa)
    end
end

end