import numpy as np

class GridWorld(object):
    def __init__(self, size, len_grid, temp_max=1) -> None:
        '''
        size: 2-D integer array. Size of the grid world.
        len_grid: Edge length of each square grid.
        '''
        self.obstacle = None # Add later
        self.heatmap = np.zeros(size)
        self.agents = []
        self.env_state = None
        self.temp_max = temp_max
        self.grid_len = grid_len

    def check(self, agent):
        '''
        Update the heatmap given the current position of agent.
        '''


    def update_heatmap(self):
        self.heatmap += 0.01
        self.heatmap[self.heatmap > self.temp_max] == self.temp_max
        for a in self.agents:
            