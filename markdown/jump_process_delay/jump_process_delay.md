# Delayed stochastic jump process
Sean L. Wu (@slwu89), 2021-12-30

## Introduction

We demonstrate how to formulate stochastic models with delay. Here, the infection process
fires at the points of a Poisson process with the same rate as the standard continuous time stochastic
SIR model. However the recovery process occurs after a deterministic delay, given by the
points of the infection process plus $\tau$, the duration of the infectious period. This example makes use of the [integrator interface](https://diffeq.sciml.ai/stable/basics/integrator/) to add in the recovery times directly into the system via a callback, while infection events are scheduled according to a rate.

## Libraries

```julia
using DifferentialEquations
using Plots
using Random
using BenchmarkTools
```




## Transitions

The infection transition is defined normally, except that it adds a time to the
`tstops` field of the integrator $\tau$ units of time from now, when the newly infected person will recover.

```julia
function infection_rate(u,p,t)
    (S,I,R) = u
    (β,c,τ) = p
    N = S+I+R
    β*c*I/N*S
end

function infection!(integrator)
    (β,c,τ) = integrator.p
    integrator.u[1] -= 1
    integrator.u[2] += 1

    # queue recovery callback
    add_tstop!(integrator, integrator.t + τ)
end

infection_jump = ConstantRateJump(infection_rate,infection!);
```




## Callbacks

The recovery process is a callback that fires according to the queued
times in `tstops`. When it fires we need to delete that element of `tstops` and
decrement `tstops_idx`. The check in the `affect!` function is because DifferentialEquations.jl
also uses `tstops` to store the final time point in the time span of the solution, so
we only allow a person to be moved from the I to R compartment if there are persons in I.

We use `reset_aggregated_jumps!` because the callback modifies the rate of the
infection jump process, so it must be recalculated after the callback fires.

```julia
function recovery_condition(u,t,integrator)
    t == integrator.tstops[1]
end

function recovery!(integrator)
    if integrator.u[2] > 0
        integrator.u[2] -= 1
        integrator.u[3] += 1
    
        reset_aggregated_jumps!(integrator)
        popfirst!(integrator.tstops)
        integrator.tstops_idx -= 1
    end
end

recovery_callback = DiscreteCallback(recovery_condition, recovery!, save_positions = (false, false))
```




We must also code a callback that will fire when the initial 10 infectives recover. Because the infectious
period is deterministic, we use a `DiscreteCallback` that fires at time $\tau$.

```julia
function affect_initial_recovery!(integrator)
    integrator.u[2] -= u0[2]
    integrator.u[3] += u0[2]

    reset_aggregated_jumps!(integrator)
end

cb_initial_recovery = DiscreteCallback((u,t,integrator) -> t == p[3], affect_initial_recovery!)
```




## Time domain

```julia
tmax = 40.0
tspan = (0.0,tmax);
```




For plotting, we can also define a separate time series.

```julia
δt = 0.1
t = 0:δt:tmax;
```




## Initial conditions

```julia
u0 = [990,10,0]; # S,I,R
```




## Parameter values

To keep the simulations broadly comparable, the fixed infectious period `τ` is `1/γ` from the other tutorials.

```julia
p = [0.05,10.0,4.0]; # β,c,τ
```




## Random number seed

We set a random number seed for reproducibility.

```julia
Random.seed!(1234);
```




## Running the model

Running this model involves:

- Setting up the problem as a `DiscreteProblem`;
- Adding the jumps and setting the algorithm using `JumpProblem`; and
- Running the model, specifying `SSAStepper()`

```julia
prob = DiscreteProblem(u0,tspan,p);
```


```julia
prob_jump = JumpProblem(prob, Direct(), infection_jump);
```


```julia
sol_jump = solve(prob_jump, SSAStepper(), callback = CallbackSet(cb_initial_recovery, recovery_callback), tstops = [p[3]]);
```




## Post-processing

In order to get output comparable across implementations, we output the model at a fixed set of times.

```julia
out_jump = sol_jump(t);
```




## Plotting

We can now plot the results.

```julia
plot(
    out_jump,
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number"
)
```

![](figures/jump_process_delay_14_1.png)



## Notes

As an alternative to using a callback, we could manually add `tstops` to the integrator, as below.

```julia
integrator = init(prob_jump,SSAStepper(), callback = recovery_callback);
for i in 1:10
	add_tstop!(integrator, integrator.t + p[3])
end
solve!(integrator)
sol_jump2 = integrator.sol
```

```
retcode: Default
Interpolation: Piecewise constant interpolation
t: 1524-element Vector{Float64}:
  0.0
  0.1257438866275816
  0.17996023375269485
  0.21024105301749613
  0.2577318665841408
  0.2992737043509679
  0.3787932286751834
  0.5210240350413188
  0.6070859730904811
  0.6184008311350182
  ⋮
 23.626384894467268
 23.669055333405325
 24.132071045440973
 24.93542233343815
 25.003740397701982
 25.6119510740526
 26.98861863907332
 27.626384894467268
 40.0
u: 1524-element Vector{Vector{Int64}}:
 [990, 10, 0]
 [989, 11, 0]
 [988, 12, 0]
 [987, 13, 0]
 [986, 14, 0]
 [985, 15, 0]
 [984, 16, 0]
 [983, 17, 0]
 [982, 18, 0]
 [981, 19, 0]
 ⋮
 [234, 7, 759]
 [234, 7, 759]
 [234, 6, 760]
 [234, 5, 761]
 [234, 4, 762]
 [234, 3, 763]
 [234, 2, 764]
 [234, 1, 765]
 [234, 0, 766]
```





## Benchmarking

```julia
@benchmark solve(prob_jump, SSAStepper(), callback = CallbackSet(cb_initial_recovery, recovery_callback), tstops = [p[3]]);
```
