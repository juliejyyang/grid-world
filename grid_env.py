import numpy as np

class Gridworld:
    def __init__(self, size=5 ):
        # env for a grid of shape size*size, we store the current agent positon (state) and goal state
        self.size = size
        self.goal_state = (size - 1, size - 1)
        self.state = None
        self.action_space = {0, 1, 2, 3} # possible actions (up, down, left, right)

    def reset(self):
        self.state = (0,0)
        return self.state

    def allowedActions(self, r, c):
        # return the set of allowed actions from state (r,c ) based on the grid's boundaries
        actions = [] 
        if r > 0: actions.append(0)
        if r < self.size - 1: actions.append(1)
        if c > 0: actions.append(2)
        if c < self.size - 1: actions.append(3)
        return actions

    def nextState(self, action, r, c):
        # given an action and current state (r,c), return the next state distribution
        if action == 0: # up
            return (r - 1, c)
        elif action == 1: # down
            return (r + 1, c)
        elif action == 2: # left
            return (r, c - 1)
        elif action == 3: # right
            return (r, c + 1)

    def reward(self, r, c, action, next_state):
        return 1.0 if next_state == self.goal_state else 0.0

    

def main():
    env = gridworld(size=5)
    env.reset()

    for i in range(50):
        env.evaluatePolicy()
        env.updatePolicy()
        
    # check that the policy is valid (sums to 1 over actions for each state)
    per_state_sums = env.P.sum(axis=0)  # sum over the action axis
    print(per_state_sums)

    print("Value function:")
    print(env.V)
    print("Policy (action probabilities):")
    print(env.P)

if __name__ == "__main__":
    main()




