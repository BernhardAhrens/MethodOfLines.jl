"""
    differentiable_problem(pdesys, disc::MOLFiniteDifference; kwargs...) -> (prob, θ0)

Discretize `pdesys` with `disc` and return a plain `ODEProblem` `prob` whose parameters
are exposed as a flat tunable `Vector`, together with the initial tunable vector `θ0`.

The problem returned by [`discretize`](@ref) carries `MOLMetadata` and stores its
parameters as `ModelingToolkit`'s `MTKParameters`. Neither the metadata-carrying solution
wrapper nor `remake`/`MTKParameters` are currently compatible with reverse-mode automatic
differentiation, so differentiating a solve with respect to the PDE parameters via Zygote
or Enzyme fails (see the discussion in
<https://github.com/SciML/MethodOfLines.jl/issues/309> and
<https://github.com/SciML/SciMLSensitivity.jl/issues/683>).

`differentiable_problem` wraps the generated right-hand side so that:

  * `prob.p` is an ordinary `Vector` holding the tunable parameters (`θ0`), and
  * `solve(prob, alg; p = θ, ...)` is differentiable with respect to `θ` using
    `ForwardDiff`, `Zygote` (through `SciMLSensitivity`) and `Enzyme`.

The tunable parameters follow `SciMLStructures.canonicalize(Tunable(), prob.p)` ordering
for the discretized system.

!!! note
    The returned solution is the raw discretized state (an `ODESolution` over the
    discretized unknowns), not a `PDETimeSeriesSolution`. This is what makes it
    AD-compatible; it also means the convenient `sol[u(t, x)]` grid indexing is not
    available on it. Use this problem for gradient-based tasks (parameter estimation,
    sensitivity analysis); use [`discretize`](@ref) for ordinary forward solves where the
    gridded solution interface is desired.

# Example

```julia
using MethodOfLines, ModelingToolkit, DomainSets, OrdinaryDiffEq, SciMLSensitivity, Zygote

prob, θ0 = differentiable_problem(pdesys, MOLFiniteDifference([x => dx], t))
predict(θ) = Array(solve(prob, Tsit5(); p = θ, saveat = 0.1))
loss(θ) = sum(abs2, predict(θ) .- data)
Zygote.gradient(loss, θ0)
```
"""
function differentiable_problem(
        pdesys::PDESystem, disc::MOLFiniteDifference; kwargs...
    )
    sys, tspan = symbolic_discretize(pdesys, disc)
    tspan === nothing && throw(ArgumentError(
        "`differentiable_problem` requires a time-dependent system; the discretized " *
        "system is stationary (a `NonlinearProblem`)."
    ))
    simpsys = mtkcompile(sys)
    molprob = ODEProblem(simpsys, nothing, tspan; disc.kwargs..., kwargs...)

    p0 = molprob.p
    θ0 = collect(SciMLStructures.canonicalize(SciMLStructures.Tunable(), p0)[1])
    u0 = molprob.u0
    molf = molprob.f

    # Re-parameterize the generated RHS by a plain tunable vector. `SciMLStructures.replace`
    # rebuilds the `MTKParameters` from `θ` in a way that is transparent to reverse-mode AD,
    # so the closure - and hence `solve(prob; p = θ)` - is differentiable w.r.t. `θ`.
    rhs! = let molf = molf, p0 = p0
        (du, u, θ, t) -> begin
            molf(du, u, SciMLStructures.replace(SciMLStructures.Tunable(), p0, θ), t)
            return nothing
        end
    end

    odefun = ODEFunction{true}(rhs!; mass_matrix = molf.mass_matrix)
    prob = ODEProblem(odefun, u0, tspan, θ0)
    return prob, θ0
end
