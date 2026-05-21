from grid_env import Gridworld
from windy_grid_env import WindyGridworld
from agent import Agent

def main():
    '''# regular gridworld env
    env = Gridworld(size=5)
    env.reset()

    agent = Agent(env, gamma = 0.5)
    agent.train(iterations=50)

    print("Value function:")
    print(agent.V)
    print("Policy (action probabilities):")
    print(agent.P)'''

    # windy grid world env
    windy_env = WindyGridworld(rows=7, cols=10)
    windy_env.reset()
    windy_agent = Agent(windy_env, gamma = 1)
    windy_agent.train(iterations=50)
    print("Windy Gridworld - Value function:")
    print(windy_agent.V)
    print("Windy Gridworld - Policy (action probabilities):")
    print(windy_agent.P)

if __name__ == "__main__":
    main()

