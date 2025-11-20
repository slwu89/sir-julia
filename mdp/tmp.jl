import POMDPs
using POMDPTools
using POMDPs: POMDP
using MCTS

using Distributions

struct state
    inventory::Int
    transit::Vector{Int}
    demand::Int
    t::Int # timestep
end

struct action
    order::Int
    dispense::Int
end

struct LogisticsMDP <: POMDPs.MDP{state, action}    
    s0::state # initial state
    γ::Float64 # discount factor
    time_horizon::Int # final time step
    # "parameters"
    order_sizes::Vector{Int} # possible order sizes
    demand_intensity::Vector{Float64} # intensity values for demand process
    storage_cost::Float64
    order_cost::Float64
    order_size_cost::Float64
    wastage_cost::Float64
    stockout_cost::Float64
end

function POMDPs.actions(mdp::LogisticsMDP, s::state)

    # always dispense as much as possible to fulfill demand
    d = min(s.inventory, s.demand)

    # we may order any of the valid size options
    return [action(o, d) for o in mdp.order_sizes]
end

function POMDPs.transition(m::LogisticsMDP, s::state, a::action)
    # movement of in-transit products
    transit_new = similar(s.transit)    
    arrived = s.transit[end]
    for k in length(s.transit):-1:2
        transit_new[k] = s.transit[k-1]
    end
    transit_new[1] = a.order

    # update inventory
    inventory_new = s.inventory + arrived - a.dispense

    # demand is where the stochasticity resides
    ImplicitDistribution() do rng
        demand = rand(rng, Poisson(m.demand_intensity[s.t+1]))
        return state(inventory_new, transit_new, demand, s.t+1)
    end
end

function POMDPs.reward(m::LogisticsMDP, s::state, a::action, s′::state)
    cost = 0.
    lost = s.demand - a.dispense
    cost += lost * m.stockout_cost
    cost += a.order * m.order_size_cost
    if a.order > 0
        cost += m.order_cost
    end
    if POMDPs.isterminal(m, s′)
        cost += s′.inventory * m.wastage_cost
    else
        cost += s′.inventory * m.storage_cost
    end
    return -cost
end

POMDPs.isterminal(m::LogisticsMDP, s::state) = s.t == m.time_horizon
POMDPs.discount(m::LogisticsMDP) = m.γ
POMDPs.initialstate(m::LogisticsMDP) = Deterministic(m.s0)

# demand (patient arrivals) is a Poisson process, but with a known intensity function, given here
# it linearly increases to a peak, then linearly decreases to zero
shipping_delay = 3 # order is added to inventory and ready to dispense k days post-order
time_horizon = 50

demand_λ = zeros(time_horizon)
demand_midpoint = Int(floor((time_horizon-shipping_delay)/2))
demand_λ[shipping_delay+1:demand_midpoint] = range(0.1, 10, length(shipping_delay+1:demand_midpoint))
demand_λ[demand_midpoint+1:end] = range(9.9, 0, length(demand_midpoint+1:time_horizon))

# set up the MDP + solver
storage_cost = 1 # storage cost per unit per day
order_cost = 10 # cost to place an order (regardless of size)
order_size_cost = 2 # cost per unit to order
wastage_cost = 50 # cost per unit of disposing of excess inventory at end of time horizon
stockout_cost = 200 # cost per unit of demand that cannot be fulfilled

# s0 = state(0, zeros(Int, shipping_delay), demand_λ[1], 1)
# inventory, transit, demand, t
s0 = state(0, zeros(Int, shipping_delay-1), demand_λ[1], 1)

logistics_mdp = LogisticsMDP(
    s0, # initial state
    0.95, # discount factor
    time_horizon, # final time step
    [1,5,10,20,25,50,100], # possible order sizes
    demand_λ, # intensity values for demand process
    storage_cost,
    order_cost,
    order_size_cost,
    wastage_cost,
    stockout_cost
)

solver = DPWSolver()

planner = solve(solver, logistics_mdp)
# a, ai = action_info(planner, logistics_mdp.s0)

for (s, a, sp, r) in stepthrough(logistics_mdp, planner, "s,a,sp,r", max_steps=10)
    println("in state $s")
    println("took action $a")
    println("arriving in new state $sp")
    println("received reward $r")
    println("")
end

sims_out = POMDPs.simulate(HistoryRecorder(max_steps=time_horizon), logistics_mdp, planner)