import numpy as np

class Gridworld:
    def __init__(self, rows = 5, cols = 5, goal_row = 4, goal_col = 4, start_row = 0, start_col = 0):
        # env for a grid of shape size*size, we store the current agent positon (state) and goal state
        self.rows = rows
        self.cols = cols
        self.goal_state = (goal_row, goal_col)
        self.start_row = start_row
        self.start_col = start_col
        self.state = None
        self.action_space = {0, 1, 2, 3} # possible actions (up, down, left, right)

    def reset(self):
        self.state = (self.start_row, self.start_col)
        return self.state

    def allowedActions(self, r, c):
        return [0, 1, 2, 3] # all actions are allowed in all states, the nextState function will handle the boundaries

    def nextState(self, action, r, c):
        # given an action and current state (r,c), return the next state distribution
        if action == 0: # up
            return (max(r - 1, 0), c)
        elif action == 1: # down
            return (min(r + 1, self.rows - 1), c)
        elif action == 2: # left
            return (r, max(c - 1, 0))
        elif action == 3: # right
            return (r, min(c + 1, self.cols - 1))

    def reward(self, r, c, action, next_state):
        return 0.0 if next_state == self.goal_state else -1.0

    






