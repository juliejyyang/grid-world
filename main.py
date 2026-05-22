from grid_env import Gridworld
from windy_grid_env import WindyGridworld
from agent import Agent
from windy_sarsa_agent import WindySarsaAgent
from visualization import visualize

def main():
    # regular gridworld env
    env = Gridworld()
    env.reset()
    agent = Agent(env, gamma = 0.5)
    agent.train(iterations=50)
    visualize(agent, title="Gridworld after policy iteration")
    
    # windy grid world env
    windy_env = WindyGridworld(rows=7, cols=10)
    windy_env.reset()
    windy_agent = Agent(windy_env, gamma = 1)
    windy_agent.train(iterations=50)
    visualize(windy_agent, title="Windy Gridworld after policy iteration")

    # SARSA agent on windy grid world
    windy_env = WindyGridworld(rows=7, cols=10)
    windy_env.reset()
    windy_sarsa_agent = WindySarsaAgent(windy_env, step_size=0.5, epsilon=0.1, gamma=1)
    windy_sarsa_agent.train(episodes=1000)
    visualize(windy_sarsa_agent, title="Windy Gridworld after SARSA learning")

if __name__ == "__main__":
    main()

