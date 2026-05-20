from grid_env import Gridworld
from agent import Agent

def main():
    env = Gridworld(size=5)
    env.reset()

    agent = Agent(env, gamma = 0.5)
    agent.train(iterations=50)

    print("Value function:")
    print(agent.V)
    print("Policy (action probabilities):")
    print(agent.P)

if __name__ == "__main__":
    main()

    