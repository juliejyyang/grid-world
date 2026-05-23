import numpy as np
from grid_env import Gridworld

class WindySarsaAgent:
    def __init__(self, env, step_size=0.5, epsilon = 0.1, gamma = 1):
        self.env = env
        self.step_size = step_size
        self.epsilon = epsilon
        self.gamma = gamma

        #initialize Q(s,a) with zeros, one entry per (action, state(r,c)) pair, initialize the goal state to have 0 value for all actions since it's terminal
        self.Q = np.zeros((len(env.action_space), env.rows, env.cols))

    def chooseAction(self, r, c):
        # choose an action from state (r,c) using epsilon-greedy exploration strategy
        if np.random.rand() < self.epsilon:
            #explore: choose a random action from the allowed actions
            possible_actions = self.env.allowedActions(r, c)
            return np.random.choice(possible_actions)

        else:
            # exploit: choose the action with the highest Q value from the allowed actions
            possible_actions = self.env.allowedActions(r, c)
            q_values = [self.Q[action][r][c] for action in possible_actions]
            max_q = max(q_values)
            max_actions = [action for action in possible_actions if self.Q[action][r][c] == max_q]
            return np.random.choice(max_actions) # break ties randomly

    def train(self, episodes = 1000):
        for episode in range(episodes):
            r, c = self.env.reset() 
            action = self.chooseAction(r, c)

            while (r, c) != self.env.goal_state:
                next_r, next_c = self.env.nextState(action, r, c)
                reward = self.env.reward((next_r, next_c))
                next_action = self.chooseAction(next_r, next_c)
                # update Q value using the SARSA update rule
                self.Q[action][r][c] += self.step_size * (reward + self.gamma * self.Q[next_action][next_r][next_c] - self.Q[action][r][c])
                r, c, action = next_r, next_c, next_action 

        # calculate the value function from the learned Q values
        self.V = self.Q.max(axis=0)

        # derive the policy from the learned Q values
        self.P = np.zeros_like(self.Q)
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                best = self.Q[:, r, c].max()
                ties = [a for a in self.env.action_space if self.Q[a, r, c] == best]
                for a in ties:
                    self.P[a, r, c] = 1.0 / len(ties)
        