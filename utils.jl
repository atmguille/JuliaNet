function ArgParse.parse_item(::Type{Vector{Int64}}, x::AbstractString)
    # Index to skip brackets at the beginning and end
    return map((a) -> parse(Int64, a), split(x[2:length(x)-1], ","))
end
