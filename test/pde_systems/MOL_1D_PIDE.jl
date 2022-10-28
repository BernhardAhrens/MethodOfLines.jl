
using NeuralPDE, Flux, MethodOfLines, ModelingToolkit, OrdinaryDiffEq, Optimization, OptimizationOptimJL, DomainSets, Plots
import ModelingToolkit: Interval, infimum, supremum

# Tests
#@testset "Test PIDE cumuSum(t,x) ~ Ix(u(t,x)), Dt(u(t, x)) + 2 * u(t, x) + 5 * Dx(cumuSum(t, x)) ~ 1" begin
    @parameters t, x
    @variables u(..) cumuSum(..)
    Dt = Differential(t)
    Dx = Differential(x)
    xmin = 0.0
    xmax = 2.0

    Ix = Integral(x in DomainSets.ClosedInterval(xmin, x)) # basically cumulative sum from 0 to x

    eq = [
        cumuSum(t,x) ~ Ix(u(t,x)), # with Integral from xmin to x: runs with NeuralPDE.jl, but not MethodOfLines.jl
        #cumuSum(t, x) ~ u(t, x), # without Integral works with NeuralPDE.jl and MethodOfLines.jl
        Dt(u(t, x)) + 2 * u(t, x) + 5 * Dx(cumuSum(t, x)) ~ 1
    ]
    bcs = [u(0.0, x) ~ 0.0, Dx(u(t, 0.0)) ~ 0.0, Dx(u(t, 2.0)) ~ 0]

    domains = [t ∈ Interval(0.0, 2.0), x ∈ Interval(xmin, xmax)]

    @named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x), cumuSum(t, x)])

    # solve with with NeuralPDE.jl
    chain = Chain(Dense(2, 15, Flux.tanh), Dense(15, 15, Flux.tanh), Dense(15, 1)) |> f64
    strategy_ = GridTraining(0.1)
    discretization = PhysicsInformedNN(chain, strategy_)

    prob = NeuralPDE.discretize(pde_system, discretization)

    callback = function (p, l)
        println("Current loss is: $l")
        return false
    end
    res = Optimization.solve(prob, BFGS(); callback=callback, maxiters=100)

    ts, xs = [infimum(d.domain):0.1:supremum(d.domain) for d in domains]
    phi = discretization.phi
    u_predict = [[first(phi([t, x], res.u)) for t in ts] for x in xs]

    p1 = plot(ts, u_predict, legend=:inline)

    # solve with MethodOfLines
    order = 2
    discretization = MOLFiniteDifference([x => 0.1], t, approx_order=order, grid_align=center_align)

    prob = MethodOfLines.discretize(pde_system, discretization)
    code = ODEFunctionExpr(prob)

    sol = solve(prob, QNDF(), saveat=0.1)


    function generate_code(pdesys::PDESystem, discretization::MethodOfLines.MOLFiniteDifference, filename="code.jl")
        code = ODEFunctionExpr(pdesys, discretization)
        rm(filename; force=true)
        open(filename, "a") do io
            println(io, code)
        end
    end

    generate_code(pde_system, discretization, "/User/homes/bahrens/minerva_bahrens/projects/hyco/code_generated.jl")

    code = ODEFunctionExpr(pde_system, discretization)

    sol.u

    p2 = plot(sol[u(t, x)], legend=:inline)

    plot(p1, p2)

    # Test against NeuralPDE solution
    for i in 1:length(t_sol)
        @test all(isapprox.(u_exact(x_sol, t_sol[i]), solu[i, :], atol=0.01))
        @test all(isapprox.(v_exact(x_sol, t_sol[i]), solv[i, :], atol=0.01))
    end
#end