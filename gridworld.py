import numpy as np

class gridworld:
    def __init__(self, size=5, gamma=0.9):
        # env for a grid of shape size*size, we store the current agent positon (state) and goal state
        self.size = size
        self.state = None
        self.goal_state = (size - 1, size - 1)
        self.action_space = {0, 1, 2, 3} # possible actions (up, down, left, right)
        self.gamma = gamma
        self.max_steps = size * 3 # reasonable limit for episode length

        # value function, one cell per state
        self.V = np.zeros((self.size, self.size))

        # policy, one per (action, state(r, c)) pair 
        self.P = np.zeros((len(self.action_space), self.size, self.size))

        # initialize the policy to be uniform (1/possible_actions) over allowed actions in each state
        for r in range(self.size):
            for c in range(self.size):
                possible_actions = self.allowedActions(r, c)
                for action in possible_actions:
                    self.P[action][r][c] = 1.0 / len(possible_actions) if len(possible_actions) > 0 else 0.0

    def allowedActions(self, r, c):
        # return the set of allowed actions from state (r,c ) based on the grid's boundaries
        actions = [] 
        if r > 0: actions.append(0)
        if r < self.size - 1: actions.append(1)
        if c > 0: actions.append(2)
        if c < self.size - 1: actions.append(3)
        return actions

    def reset(self):
        # reset the env to an initial state, by default the agent starts at (0,0)
        # returns the initial state
        self.state = (0,0)
        self.step_count = 0

        return self.state 

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

    def evaluatePolicy(self):
        Vnew = np.zeros((self.size, self.size))
        for r in range(self.size):
            for c in range(self.size):
                # the goal is terminal so we can skip it since its value is always 0
                if (r, c) == self.goal_state:
                    continue
                v = 0.0
                possible_actions = self.allowedActions(r, c)  
                for action in possible_actions:
                    prob = self.P[action][r][c] # transition probability for action from state (r,c)
                    next_state = self.nextState(action, r, c) # look up the next state distribution for this action 
                    reward = self.reward(r, c, action, next_state) # compute the reward for taking this action from state (r,c)
                    v += prob * (reward + self.gamma * self.V[next_state]) # update the value function using the Bellman equation 
                Vnew[r][c] = v
        self.V = Vnew # update the value function 

    def updatePolicy(self):
        for r in range(self.size):
            for c in range(self.size):
                if (r, c) == self.goal_state:
                    continue
                possible_actions = self.allowedActions(r, c)

                # compute the value of taking each allowed action from the state (r, c)
                vmax = None
                nmax = None
                vs = [] 
                for action in possible_actions:
                    # compute the value of taking this action from state (r,c) using the current value function
                    next_state = self.nextState(action, r, c)
                    reward = self.reward(r, c, action, next_state)
                    v = reward + self.gamma * self.V[next_state] 
                    vs.append(v)
                    if vmax is None or v > vmax:
                        vmax = v
                        nmax = 1
                    elif v == vmax:
                        nmax += 1 
                # update the policy to be uniform over the actions that achieve the maximum value, enumerate them
                for i, action in enumerate(possible_actions):
                    self.P[action][r][c] = 1.0 / nmax if vs[i] == vmax else 0.0 

def main():
    env = gridworld(size=5)
    env.reset()

    for i in range(50):
        env.evaluatePolicy()
        env.updatePolicy()

    print("Value function:")
    print(env.V)
    print("Policy (action probabilities):")
    print(env.P)

if __name__ == "__main__":
    main()




