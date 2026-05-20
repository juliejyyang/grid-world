import numpy as np

class Agent:
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        self.V = np.zeros((env.size, env.size))  # value function, one cell per state
        self.P = np.zeros((len(env.action_space), env.size, env.size))  # policy, one per (action, state(r, c)) pair
        
        # initialize the policy to be uniform (1/possible_actions) over allowed actions in each state
        for r in range(self.env.size):
            for c in range(self.env.size):
                possible_actions = self.env.allowedActions(r, c)
                for action in possible_actions:
                    self.P[action][r][c] = 1.0 / len(possible_actions) if len(possible_actions) > 0 else 0.0

    def evaluatePolicy(self):
        Vnew = np.zeros((self.env.size, self.env.size))
        for r in range(self.env.size):
            for c in range(self.env.size):
                # the goal is terminal so we can skip it since its value is always 0
                if (r, c) == self.env.goal_state:
                    continue
                v = 0.0
                possible_actions = self.env.allowedActions(r, c)  
                for action in possible_actions:
                    prob = self.P[action][r][c] # transition probability for action from state (r,c)
                    next_state = self.env.nextState(action, r, c) # look up the next state distribution for this action 
                    reward = self.env.reward(r, c, action, next_state) # compute the reward for taking this action from state (r,c)
                    v += prob * (reward + self.gamma * self.V[next_state]) # update the value function using the Bellman equation 
                Vnew[r][c] = v
        self.V = Vnew # update the value function 

    def updatePolicy(self):
        for r in range(self.env.size):
            for c in range(self.env.size):
                if (r, c) == self.env.goal_state:
                    continue
                possible_actions = self.env.allowedActions(r, c)

                # compute the value of taking each allowed action from the state (r, c)
                vmax = None
                nmax = None
                vs = [] 
                for action in possible_actions:
                    # compute the value of taking this action from state (r,c) using the current value function
                    next_state = self.env.nextState(action, r, c)
                    reward = self.env.reward(r, c, action, next_state)
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

    def train(self, iterations = 50):
        for i in range(iterations):
            self.evaluatePolicy()
            self.updatePolicy()



    
