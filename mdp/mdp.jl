import POMDPs
using POMDPTools
using POMDPs: POMDP
using MCTS
using Distributions, Plots

@inline function rate_to_prob(r::T, t::T) where {T<:AbstractFloat}
    1-exp(-r*t)
end;

struct state
    S::Float64
    I::Float64
    C::Float64
    υ_sum::Float64 # cumulative intervention
    t::Int # timestep
end

action = Float64

υ_max # maximum intervention


struct LockdownMDP <: POMDPs{state, action}
    s0::state # initial state
    discount::Float64 # discount of future rewards
    nsteps::Int
    # parameters
    β::Float64
    c::Float64
    γ::Float64
    δt::Float64
    # control parameters
    υ_total::Float64 # maximum cost
    𝒜::Vector{Float64} # action space
end

function POMDPs.actions(m::LockdownMDP, s::state)
    if state.υ_sum >= m.υ_total
        return [0.]
    else
        return m.𝒜
    end    
end

function POMDPs.transition(m::LockdownMDP, s::state, a::action)
    S, I, R = (s.S, s.I, s.R)
    N = S + I + R
    s_i_prob = rate_to_prob((1-a)*β*c*I/N, m.δt)
    i_r_prob = rate_to_prob(γ, m.δt)
    s_i = rand(Binomial(S, s_i_prob))
    i_r = rand(Binomial(I, i_r_prob))
    return state(S-s_i, I+s_i-i_r, R+i_r)
end

function POMDPs.reward(m::LogisticsMDP, s::state, a::action, s′::state)
    return -s′.R
end