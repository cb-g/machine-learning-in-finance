using Markdown
using InteractiveUtils
using ReinforcementLearning
using Plots
using Flux
using Statistics

begin
	const NX = 4
	const NY = 12
	const Start = CartesianIndex(1, 1)
	const Goal = CartesianIndex(1, 12)
	const UDZ = [
        CartesianIndex(-1, 0),  # up
        CartesianIndex(1, 0),   # down
        CartesianIndex(0, 0),  # zero
	]
	const LinearInds = LinearIndices((NX, NY))
end

function iscliff(p::CartesianIndex{2})
    x, y = Tuple(p)
    x == 4 && y > 1 && y < NY
end

heatmap((!iscliff).(CartesianIndices((NX, NY))); yflip = true)

begin
	Base.@kwdef mutable struct CliffWalkingEnv <: AbstractEnv
		position::CartesianIndex{2} = Start
	end
	function (env::CliffWalkingEnv)(a::Int)
		x, y = Tuple(env.position + UDZ[a])
		env.position = CartesianIndex(min(max(x, 1), NX), min(max(y, 1), NY))
	end
end

RLBase.state(env::CliffWalkingEnv) = LinearInds[env.position]

RLBase.state_space(env::CliffWalkingEnv) = Base.OneTo(length(LinearInds))

RLBase.action_space(env::CliffWalkingEnv) = Base.OneTo(length(UDZ))

RLBase.reward(env::CliffWalkingEnv) = env.position == Goal ? 0.0 : (iscliff(env.position) ? -100.0 : -1.0)

RLBase.is_terminated(env::CliffWalkingEnv) = env.position == Goal || iscliff(env.position)

RLBase.reset!(env::CliffWalkingEnv) = env.position = Start

world = CliffWalkingEnv()

begin
	NS = length(state_space(world))
	NA = length(action_space(world))
end

create_agent(α, method) = Agent(
	policy = QBasedPolicy(
		learner=TDLearner(
			approximator=TabularQApproximator(
				;n_state=NS,
				n_action=NA,
				opt=Descent(α),
			),
			method=method,
			γ=1.0,
			n=0
		),
		explorer=EpsilonGreedyExplorer(0.1)
	),
	trajectory=VectorSARTTrajectory()	
)

function repeated_run(α, method, N, n_episode, is_mean=true)
	env = CliffWalkingEnv()
	rewards = []
	for _ in 1:N
		h = TotalRewardPerEpisode(;is_display_on_exit=false)
		run(
			create_agent(α, method),
			env, 
			StopAfterEpisode(n_episode;is_show_progress=false),
			h
		)
		push!(rewards, is_mean ? mean(h.rewards) : h.rewards)
	end
	mean(rewards)
end

α = 0.001
N = 500
n_episode = 500

begin
	p = plot(legend=:bottomright, xlabel="Episodes", ylabel="Sum of rewards during episode")
	plot!(p, repeated_run(α, :SARS, N, n_episode, false), label="QLearning")
	plot!(p, repeated_run(α, :SARSA, N, n_episode, false), label="SARSA")
	p
end

# begin
# 	A = 0.1:0.05:0.95
# 	fig_9_4 = plot(;legend=:bottomright, xlabel="α", ylabel="Sum of rewards per episode")

# 	plot!(fig_9_4, A, [repeated_run(α, :SARS, 100, 100) for α in A], linestyle=:dash ,markershape=:rect, label="Interim Q")
# 	plot!(fig_9_4, A, [repeated_run(α, :SARSA, 100, 100) for α in A], linestyle=:dash, markershape=:dtriangle, label="Interim SARSA")
# 	plot!(fig_9_4, A, [repeated_run(α, :ExpectedSARSA, 100, 100) for α in A], linestyle=:dash, markershape=:cross, label="Interim ExpectedSARSA")

# 	plot!(fig_9_4, A, [repeated_run(α, :SARS, 10, 5000) for α in A], linestyle=:solid ,markershape=:rect, label="Asymptotic interim Q")
# 	plot!(fig_9_4, A, [repeated_run(α, :SARSA, 10, 5000) for α in A], linestyle=:solid, markershape=:dtriangle, label="Asymptotic SARSA")
# 	plot!(fig_9_4, A, [repeated_run(α, :ExpectedSARSA, 10, 5000) for α in A], linestyle=:solid, markershape=:cross, label="Asymptotic ExpectedSARSA")
# 	fig_9_4
# end