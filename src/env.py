import numpy as np

class GridWorld(object):
    def __init__(self, size_world, len_grid) -> None:
        '''
        The left-up is (0, 0). Down and right are positive directions for y and x.
        size: 2-D integer array. Size of the grid world.
        len_grid: Edge length of each square grid.
        '''
        self.obstacle = None # Add later
        self.heatmap = np.zeros(size_world)
        self.agents = []
        self.env_state = None
        self.temp_max = 1
        self.len_grid = len_grid
        self.x_coord = 0.5 * len_grid + np.repeat(np.arange(size_world[1])[None, :] * len_grid, size_world[0], axis=0)
        self.y_coord = 0.5 * len_grid + np.repeat(np.arange(size_world[0])[:, None] * len_grid, size_world[1], axis=1)
        # self.center_locations = np.empty((*size_world, 2))
        # self.center_locations[:, :, 0] = x_coord
        # self.center_locations[:, :, 1] = y_coord
        

    def check(self, agent):
        '''
        Update the heatmap given the current position of one agent.
        '''
        dist = np.sqrt((self.x_coord - agent.state[0])**2 + (self.y_coord - agent.state[1])**2)
        checked_grid = dist < agent.r_s
        self.heatmap[checked_grid] = 0


    def update_heatmap(self):
        self.heatmap += 0.01
        self.heatmap[self.heatmap > self.temp_max] == self.temp_max
        for a in self.agents:
            self.check(a)

    
    def add_agents(self, agents):
        self.agents = agents


if __name__=='__main__':
    world = GridWorld((10, 10), 1)
    class Agent(object):
        def __init__(self, init_state) -> None:
            self.state = init_state
            self.r_s = 2.0
    
    agents = [Agent(np.array([0., 0.]))]

    world.add_agents(agents)
    for _ in range(10):
        print(world.heatmap)
        world.update_heatmap()
        world.agents[0].state += 1.0
