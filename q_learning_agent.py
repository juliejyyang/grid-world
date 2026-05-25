from windy_sarsa_agent import WindySarsaAgent
from numpy import np

class qLearningAgent(WindySarsaAgent):
    def __init__(self, env, step_size=0.5, epsilon=0.1, gamma=0.9):
        super().__init__(env, step_size, epsilon, gamma)
        self.Q = np.zeros((len(env.action_space), env.rows, env.cols))

    def train(self, episodes=1000):
        for episode in range(episodes):
            r, c = self.env.reset()
            action = self.chooseAction(r, c)

            while (r, c) != self.env.goal_state:
                next_r, next_c = self.env.nextState(action, r, c)
                reward = self.env.reward((next_r, next_c))

                #update Q value using Q-learning rule
                best_next_q = self.Q[:, next_r, next_c].max()
                self.Q[action][r][c] += self.step_size * (reward + self.gamma * best_next_q - self.Q[action][r][c])
                r, c = next_r, next_c 