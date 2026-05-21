import numpy as np
from grid_env import Gridworld

# strength of wind is given below each column 
wind_strength = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])

class WindyGridworld (Gridworld):
    def __init__(self, rows = 7, cols = 10):
        super().__init__(rows, cols)

    def nextState(self, action, r, c):
        if action == 0: #up
            return (max(r - 1 - wind_strength[c], 0), c)
        elif action == 1: #down 
            return (max(min(r + 1 - wind_strength[c], self.rows - 1), 0), c)
        elif action == 2: #left
            return (max(r - wind_strength[c], 0), max(0, c - 1))
        elif action == 3: #right
            return (max(r - wind_strength[c], 0), min(c + 1, self.cols - 1))