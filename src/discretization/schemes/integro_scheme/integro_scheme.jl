# first steps, starting from here: https://docs.sciml.ai/MethodOfLines/stable/devnotes/

#TODO: Add handling for cases where II is close to the boundaries
#TODO: Handle periodic boundary conditions
#TODO: Handle nonuniformly discretized `x`
function piecewise_sum(II::CartesianIndex, s::DiscreteSpace, u, x)
#=     # Get which place `x` appears in `u`'s arguments
    j = x2i(s, u, x)

    # Get a CartesianIndex of unit length that points in the direction of `x` e.g. CartesianIndex((1, 0, 0))
    I1 = unitindex(ndims(u, s), j) 

    discu = s.discvars[u]
    expr = sum(discu[first:II]*s.dx[first:x])/x

    return expr =#
        # Get which place `x` appears in `u`'s arguments
        j = x2i(s, u, x)

        # Get a CartesianIndex of unit length that points in the direction of `x` e.g. CartesianIndex((1, 0, 0))
        I1 = unitindex(ndims(u, s), j) 
    
        discu = s.discvars[u]
        expr = (discu[II + I1] - discu[II - I1])/s.dx[x]
    
        return expr
end

# Note that indexmap is used along with the function `Idx` to create an equivalent index for the discrete form of `u`,
# which may have a different number of dimensions to `II`
@inline  function generate_integrate(II::CartesianIndex, s::DiscreteSpace, terms, indexmap, depvars) 
    #(II::CartesianIndex, s::DiscreteSpace, depvars, derivweights::DifferentialDiscretizer, pmap, indexmap, terms)
    #(II::CartesianIndex, s::DiscreteSpace, terms, indexmap, depvars) 
#=     rules = [[@rule Integrate(x)(u) => piecewise_sum(Idx(II, s, u, indexmap), s, u, x) for x in params(u, x)] for u in depvars]

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
    return rule_pairs =#

#=     central_ufunc(u, I, x) = s.discvars[u][I]
    return reduce(vcat, [reduce(vcat, [[Integrate(x)(u) => piecewise_sum(derivweights.map[Integrate(x)], Idx(II, s, u, indexmap), s, pmap.map[operation(u)][x], (x2i(s, u, x), x), u, central_ufunc) for d in (
        let orders = derivweights.orders[x]
            orders[iseven.(orders)]
        end
    )] for x in params(u, s)]) for u in depvars]) =#

    #central_ufunc(u, I, x) = s.discvars[u][I]
    rules = [[@rule Integral(x, 0..x)(u) => piecewise_sum(Idx(II, s, u, indexmap), s, u, x) for x in params(u, s)] for u in depvars]

    rules = reduce(vcat, rules)
    println(rules)

    return(rules)
end