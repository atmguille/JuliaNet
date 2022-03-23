module RedNeuronal_pkg

using ..Capa_pkg
using ..Neurona_pkg

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


function Propagar_y_Disparar(red_neuronal::RedNeuronal)
    for index_capa in 1:(size(red_neuronal.capas,1)-1)
        Capa_pkg.Propagar(red_neuronal.capas[index_capa])
        Capa_pkg.Disparar(red_neuronal.capas[index_capa + 1])
    end
end

"""
    CrearRedAleatoria(configuracion::Vector{Int64}, peso_min::Float64, peso_max::Float64)

Crea una red neuronal aleatoria en base a una configuración
# Arguments:
- `configuracion::Vector{Int64}`: lista de números enteros que indican el número de neuronas
                                  en cada capa
- `peso_min::Float64`: mínimo valor de los pesos
- `peso_max::Float64`: máximo valor de los pesos

"""
function CrearRedAleatoria(configuracion::Vector{Int64}, peso_min::Float64, peso_max::Float64)
    red_neuronal = Crear()
    n_capas = size(configuracion, 1)
    for (capa_index, n_neuronas) in enumerate(configuracion)
        capa = Capa_pkg.Crear()

        # La primera capa no tiene función de activación
        if capa_index == 1
            for i in 1:n_neuronas
                neurona = Neurona_pkg.Crear(0.0, Neurona_pkg.Directa)
                Capa_pkg.Añadir(capa, neurona)
            end
        else
            for i in 1:n_neuronas
                neurona = Neurona_pkg.Crear(0.0, Neurona_pkg.Sigmoide)
                Capa_pkg.Añadir(capa, neurona)
            end
        end

        # La última capa no tiene sesgo
        if capa_index != n_capas
            Capa_pkg.Añadir(capa, Neurona_pkg.Crear(1.0, Neurona_pkg.Sesgo))
        end
        # La primera capa es la única sin conexión con la anterior
        if capa_index != 1
            Capa_pkg.Conectar(last(red_neuronal.capas), capa, peso_min, peso_max)
        end
        Añadir(red_neuronal, capa)
    end

    return red_neuronal
end

"""
# TODO: update documentation
    avanzar_ciclo(red::RedNeuronal, valores_entrada::Vector)

Avanza un ciclo de la red neuronal. Consiste en:
* Inicializar la capa de entrada a los valores de entrada
* Disparar todas las neuronas de la red
* Descargar las neuronas de la red (inicialización de valores de entrada a 0)
* Propagar todas las neuronas de la red

# Arguments:
- `red::RedNeuronal`: Red neuronal
- `valores_entrada::Vector`: Valores de entrada

"""
function Feedforward(red::RedNeuronal, valores_entrada::Vector{Float64})
    capa_entrada = red.capas[1]
    for i in 1:size(valores_entrada, 1)
        Neurona_pkg.Inicializar(capa_entrada.neuronas[i], valores_entrada[i])
    end
    Capa_pkg.Disparar(capa_entrada)
    Inicializar(red)
    Propagar_y_Disparar(red)
end

function Backpropagation(red::RedNeuronal, clases_verdaderas::Vector{Float64},
                         tasa_aprendizaje::Float64)
    capa_salida = last(red.capas)
    deltas = Vector{Float64}()

    for i in 1:size(clases_verdaderas, 1)
        delta = clases_verdaderas[i] - capa_salida.neuronas[i].valor_salida
        delta *= Neurona_pkg.DerivadaActivacion(capa_salida.neuronas[i])
        push!(deltas, delta)
    end

    n_capas = size(red.capas, 1)

    for (capa_index, capa) in enumerate(reverse(red.capas[1:n_capas - 1]))
       deltas_capa = Vector{Float64}()
       salidas_capa = []
        for neurona in capa.neuronas
            push!(salidas_capa, neurona.valor_salida)
            # No es necesario calcular delta para Sesgo ni para primera capa de la red
            if neurona.tipo == Neurona_pkg.Sesgo || capa_index == n_capas - 1
                continue
            end
            pesos_neurona = []
            for conexion in neurona.conexiones
                push!(pesos_neurona, conexion.peso)
            end
            delta = Neurona_pkg.DerivadaActivacion(neurona) * (deltas' * pesos_neurona)
            push!(deltas_capa, delta)
        end
        delta_pesos = tasa_aprendizaje .* (salidas_capa * deltas')
        for (neurona_index, neurona) in enumerate(capa.neuronas)
            for (conexion_index, conexion) in enumerate(neurona.conexiones)
                conexion.peso_anterior = conexion.peso
                conexion.peso += delta_pesos[neurona_index, conexion_index]
            end
        end
        deltas = deltas_capa
        end
    end


end