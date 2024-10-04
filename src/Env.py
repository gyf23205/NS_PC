import numpy as np

class GridWorld(object):
    def __init__(self, size, temp_max=1) -> None:
        self.obstacle = None # Add later
        self.heatmap = np.zeros(size)
        self.agents = []
        self.env_state = None
        self.temp_max = temp_max

    def update_heatmap(self):
        self.heatmap += 0.01
        self.heatmap[self.heatmap > self.temp_max] == self.temp_max
        # for a in self.agents:

    