import numpy as np
import matplotlib.pyplot as plt
class GridWorld(object):
    def __init__(self, size_world, len_grid, obstacles) -> None:
        '''
        The left-up is (0, 0). Down and right are positive directions for y and x.
        size: 2-D integer array. Size of the grid world.
        len_grid: Edge length of each square grid.
        '''
        self.obstacle = None # Add later
        self.heatmap = np.ones(size_world)
        self.agents = []
        self.env_state = None
        self.temp_max = 1
        self.len_grid = len_grid
        self.x_coord = 0.5 * len_grid + np.repeat(np.arange(size_world[1])[None, :] * len_grid, size_world[0], axis=0)
        self.y_coord = 0.5 * len_grid + np.repeat(np.arange(size_world[0])[:, None] * len_grid, size_world[1], axis=1)
        # fig, ax = plt.subplots()
        # im = ax.imshow(self.x_coord)
        # plt.xlim((0,40))
        # plt.ylim((0,40))
        # plt.show()
        # self.center_locations = np.empty((*size_world, 2))
        # self.center_locations[:, :, 0] = x_coord
        # self.center_locations[:, :, 1] = y_coord
        

    def check(self, agent):
        '''
        Update the heatmap given the current position of one agent.
        '''
        dist = np.sqrt((self.x_coord - agent.states[0])**2 + (self.y_coord - agent.states[1])**2)
        # fig, ax = plt.subplots()
        # im = ax.imshow(dist)
        # plt.xlim((0,80))
        # plt.ylim((0,80))
        # plt.show()
        checked_grid = dist < agent.r_s
        self.heatmap[checked_grid] = 0


    def update_heatmap(self):
        self.heatmap += 0.003
        self.heatmap[self.heatmap > self.temp_max] = self.temp_max
        for a in self.agents:
            self.check(a)

    
    def add_agents(self, agents):
        self.agents = agents

    def get_cost_max(self):
        '''
        Get the maximum value of the heatmap, it could be strict and the reward is always -max
        '''
        return -np.max(self.heatmap)
    
    def get_cost_avg(self):
        '''
        For every grid in the heatmap, reward = - (distance to the cloest agent) * heat of this grid.
        But this could be too gentle, leaving some grid not been checked at all.
        '''
        dists = []
        for agent in self.agents:
            dists.append(np.sqrt((self.x_coord - agent.states[0])**2 + (self.y_coord - agent.states[1])**2))
        dists = np.array(dists)
        idx_agent_cloest = np.argmin(dists, axis=0)
        cost = np.zeros_like(dists[0, :, :])
        # Try to get rid of the double for loop
        for i in range(cost.shape[0]):
            for j in range(cost.shape[1]):
                cost[i, j] = self.heatmap[i, j] * dists[idx_agent_cloest[i, j], i, j]

        return cost




if __name__=='__main__':
    world = GridWorld((10, 10), 1, None)
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
