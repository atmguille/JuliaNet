function ArgParse.parse_item(::Type{Vector{Int64}}, x::AbstractString)
    # Índices para evitar corchetes de inicio y fin
    return map((a) -> parse(Int64, a), split(x[2:length(x)-1], ","))
end
