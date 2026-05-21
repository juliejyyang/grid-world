import numpy as np

# strength of wind is given below each column 
wind_strength = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])

class WindyGridworld (Gridworld):
    def __init__(self, size=5):
        super().__init__(size)

    def nextState(self, action, r, c):
        if action == 0: #up
            return (max(r - 1 - wind_strength[c], 0), c)
        elif action == 1: #down 
            return (min(r + 1 - wind_strength[c], self.size - 1), c)
        elif action == 2: #left
            return (r, max(c - 1 - wind_strength[c], 0))
        elif action == 3: #right
            return (r, min(c + 1 - wind_strength[c], self.size - 1))