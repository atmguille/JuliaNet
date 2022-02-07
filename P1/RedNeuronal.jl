#=
RedNeuronal:
- Julia version: 
- Author: guille806
- Date: 2022-02-07
=#

module RedNeuronal_pkg

include("Capa.jl")
using .Capa_pkg

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
    for capa in red_neuronal.Capas
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

red = RedNeuronal_pkg.Crear()
neurona1 = RedNeuronal_pkg.Capa_pkg.Neurona_pkg.Crear(1.0, RedNeuronal_pkg.Capa_pkg.Neurona_pkg.Sesgo)
neurona2 = RedNeuronal_pkg.Capa_pkg.Neurona_pkg.Crear(1.0, RedNeuronal_pkg.Capa_pkg.Neurona_pkg.Sesgo)
RedNeuronal_pkg.Capa_pkg.Neurona_pkg.Conectar(neurona1, neurona2, 0.5)
capa = RedNeuronal_pkg.Capa_pkg.Crear()
RedNeuronal_pkg.Capa_pkg.Añadir(capa, neurona1)
RedNeuronal_pkg.Añadir(red, capa)
RedNeuronal_pkg.Disparar(red)
RedNeuronal_pkg.Propagar(red)
print(neurona1.valor_salida)
print(neurona2.valor_entrada)
RedNeuronal_pkg.Liberar(red)