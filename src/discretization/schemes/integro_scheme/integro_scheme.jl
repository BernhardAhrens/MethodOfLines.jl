# first steps, starting from here: https://docs.sciml.ai/MethodOfLines/stable/devnotes/

#TODO: Add handling for cases where II is close to the boundaries
#TODO: Handle periodic boundary conditions
#TODO: Handle nonuniformly discretized `x`
function piecewise_sum(II::CartesianIndex, s::DiscreteSpace, u, x)
    # Get which place `x` appears in `u`'s arguments
    j = x2i(s, u, x)

    # Get a CartesianIndex of unit length that points in the direction of `x` e.g. CartesianIndex((1, 0, 0))
    I1 = unitindex(ndims(u, s), j) 

    discu = s.discvars[u]
    expr = sum(discu[first:II]*s.dx[first:x])/x

    return expr
end

# Note that indexmap is used along with the function `Idx` to create an equivalent index for the discrete form of `u`,
# which may have a different number of dimensions to `II`
function generate_central_difference_rules(II::CartesianIndex, s::DiscreteSpace, terms::Vector{<:Term}, indexmap::Dict)
    rules = [[@rule Integrate(x)(u) => piecewise_sum(Idx(II, s, u, indexmap), s, u, x) for x in params(u, x)] for u in depvars]

    rules = reduce(vcat, rules)

    # Parse the rules in to pairs that can be used with `substitute`, this can be copy pasted.
    rule_pairs = []
    for t in terms
        for r in rules
            if r(t) !== nothing
                push!(rule_pairs, t => r(t))
            end
        end
    end
    return rule_pairs
end