using MethodOfLines, ModelingToolkit, DomainSets, OrdinaryDiffEq, ForwardDiff, Test

# `differentiable_problem` should build a plain `ODEProblem` whose parameters are a flat
# tunable vector, so that solving is compatible with automatic differentiation w.r.t. the
# PDE parameters. Here we check ForwardDiff against a central finite difference; Zygote and
# Enzyme are validated out-of-tree to avoid pulling heavy AD backends into the test env.

@parameters t x a0 a1
@variables u(..)
Dt = Differential(t)
Dxx = Differential(x)^2

heateq = [Dt(u(t, x)) ~ 2.0 * a0 * u(t, x) + a1 * Dxx(u(t, x))]
bcs = [u(t, 0) ~ 0.0, u(t, 10.0) ~ 0.0, u(0, x) ~ exp(-(x - 3.0)^2)]
domains = [t ∈ Interval(0.0, 0.2), x ∈ Interval(0.0, 10.0)]
@named pdesys = PDESystem(heateq, bcs, domains, [t, x], [u(t, x)], [a0 => 0.1, a1 => 0.2])

prob, θ0 = differentiable_problem(pdesys, MOLFiniteDifference([x => 0.2], t))

@test prob isa ODEProblem
@test prob.p isa AbstractVector
@test eltype(prob.p) <: Real
@test length(θ0) == 2

saveat = 0.05
loss(θ) = sum(abs2, Array(solve(prob, Tsit5(); p = θ, saveat, abstol = 1e-10, reltol = 1e-10)))

# forward solve works and is finite
@test isfinite(loss(θ0))

g_ad = ForwardDiff.gradient(loss, θ0)

# central finite-difference reference
function fd_grad(f, θ; h = 1e-6)
    g = similar(θ)
    for i in eachindex(θ)
        θp = copy(θ); θp[i] += h
        θm = copy(θ); θm[i] -= h
        g[i] = (f(θp) - f(θm)) / (2h)
    end
    return g
end
g_fd = fd_grad(loss, θ0)

@test g_ad ≈ g_fd rtol = 1e-4
@test all(isfinite, g_ad)

# a stationary system has no time variable and must be rejected
@parameters xs ys
@variables v(..)
Dxx_s = Differential(xs)^2
Dyy_s = Differential(ys)^2
laplace = [Dxx_s(v(xs, ys)) + Dyy_s(v(xs, ys)) ~ 0]
bcs_s = [v(0, ys) ~ 0.0, v(1, ys) ~ 0.0, v(xs, 0) ~ 0.0, v(xs, 1) ~ 0.0]
domains_s = [xs ∈ Interval(0.0, 1.0), ys ∈ Interval(0.0, 1.0)]
@named laplacesys = PDESystem(laplace, bcs_s, domains_s, [xs, ys], [v(xs, ys)])
@test_throws ArgumentError differentiable_problem(
    laplacesys, MOLFiniteDifference([xs => 0.2, ys => 0.2]))
