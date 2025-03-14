import numpy as np
import matplotlib.pyplot as plt


class GridWorld(object):
    def __init__(self, size_world, len_grid, heatmap, obstacles, decay=0.001, increase=0.05) -> None:
        '''
        The left-up is (0, 0). Down and right are positive directions for y and x.
        size: 2-D integer array. Size of the grid world.
        len_grid: Edge length of each square grid.
        rate: 2-D array with the same size of heatmap, indicating the changing rate of each grid.
        decay: Decaying rate of coverage level
        increase: Increasing rate of coverage level when been covered by agents. cov_lvl += increase * n_agents_covering.
        '''
        self.size_world = size_world
        self.obstacle = None # Add later
        self.heatmap = heatmap
        self.cov_lvl = np.zeros_like(self.heatmap) # Coverage level, naturally decay. Increase when a grid been checked.
        self.heatmap_pad = None
        self.agents = []
        self.env_state = None
        self.cov_max = 1
        self.heat_max = 1
        self.decay = decay
        self.increase = increase
        self.len_grid = len_grid
        self.agent_rs = None
        self.x_coord = 0.5 * len_grid + np.repeat(np.arange(size_world[1])[None, :] * len_grid, size_world[0], axis=0)
        self.y_coord = 0.5 * len_grid + np.repeat(np.arange(size_world[0])[:, None] * len_grid, size_world[1], axis=1)
        print()
      
    def _grid_agents_dist(self):
        dist = []
        for a in self.agents:
            dx = self.x_coord - a.states[0]
            dy = self.y_coord - a.states[1]
            dist.append(np.sqrt(dx**2 + dy**2))
        return dist

    def agents_dist(self):
        '''
        Get the inter-agent distance matrix. This is a bit cheating. In real experiment, this need to be replaced by a sensing based function.
        '''
        coords = np.array([a.states[0:2] for a in self.agents])
        dist = np.array([np.linalg.norm(coords[i, :] - coords, axis=1) for i in range(coords.shape[0])])
        return dist
    

    def check(self):
        '''
        Check the surrounding environemtn and return the observations. Now using square shape observing regions.
        '''
        observations = []
        grid_agent_dist = self._grid_agents_dist()
        for i, agent in enumerate(self.agents):
            # dist = np.sqrt((self.x_coord - agent.states[0])**2 + (self.y_coord - agent.states[1])**2)
            current_grid = (int((agent.states[0]+ self.agent_rs) // self.len_grid ), int((agent.states[1] + self.agent_rs) // self.len_grid))
            left, right = current_grid[0] - self.agent_rs, current_grid[0] + self.agent_rs,
            up, down = current_grid[1] - self.agent_rs, current_grid[1] + self.agent_rs
            # print(current_grid)
            # print((left, right, up, down))
            temp = np.copy(self.heatmap_pad)
            mask = grid_agent_dist[i] > self.agent_rs
            mask = np.pad(mask, np.ceil(self.agents[0].r_s), constant_values=True)
            temp[mask] = 0
            observations.append(temp[None, up:down, left:right])
            # observations.append(np.ones((1,6,6))*0.1*i)
        return observations

    def step(self):
        grid_agent_dist = self._grid_agents_dist()
        for d in grid_agent_dist:
            self.cov_lvl[d < self.agent_rs] += self.increase
        self.cov_lvl -= self.decay
        self.cov_lvl[self.cov_lvl < 0] = 0
        self.cov_lvl[self.cov_lvl > self.cov_max] = self.cov_max
        
    
    def add_agents(self, agents):
        self.agents = agents
        self.agent_rs = self.agents[0].r_s
        self.heatmap_pad = np.pad(self.heatmap, np.ceil(self.agents[0].r_s))

    def get_agent_cost(self, id):
        grid_agent_dist = self._grid_agents_dist()
        for i, a in enumerate(self.agents):
            if a.id == id:
                mask = grid_agent_dist[i] < self.agent_rs
                break
        return -np.mean(self.heatmap[mask] * self.cov_lvl[mask]) # By overlapping, each agent can get lower cost. Is this reasonable?

    def get_cost_max(self):
        '''
        Get the maximum value of the heatmap, it could be strict and the cost is always max
        '''
        return np.max(self.heatmap * (self.cov_max - self.cov_lvl))
    
    def get_cost_mean(self,thre):
        if np.mean(self.heatmap * self.cov_lvl) > thre:
            return -1
        else:
            return 0
        # return -np.mean(self.heatmap * self.cov_lvl)
    
    # def get_cost_avg(self):
    #     '''
    #     For every grid in the heatmap, reward = - (distance to the cloest agent) * heat of this grid.
    #     But this could be too gentle, leaving some grid not been checked at all.
    #     '''
    #     dists = []
    #     for agent in self.agents:
    #         dists.append(np.sqrt((self.x_coord - agent.states[0])**2 + (self.y_coord - agent.states[1])**2))
    #     dists = np.array(dists)
    #     idx_agent_cloest = np.argmin(dists, axis=0)
    #     cost = np.zeros_like(dists[0, :, :])
    #     # Try to get rid of the double for loop
    #     row, col = np.indices(self.heatmap.shape)  # row indices = column indices^T, 10*10. 
    #     cost = self.heatmap * dists[idx_agent_cloest, row, col]  # dim(dists[idx_agent_cloest]) = 10*10*10*10, dim(dists[idx_agent_cloest, row, col]) = 10*10
    #     return cost
    
    

class GridFireWorld(GridWorld):
    def __init__(self, size_world, len_grid, rate, obstacles, init_fire):
        '''
        rate: Rate of temprature decaying
        init_fire: The initial position of fire source
        '''
        super().__init__(size_world, len_grid, rate, obstacles)
        self.pos_fire = init_fire
        self.temp_min = 0


    def step_heatmap(self):
        step = np.random.choice(np.arange(0, 4), 1)
        right = np.min(self.size_world[0], self.pos_fire[0]+1)
        left = np.max(0, self.pos_fire[0]-1)
        down = np.min(self.size_world[1], self.pos_fire[0]+1)
        up = np.max(0, self.pos_fire[0]-1)
        if step == 0:
            self.pos_fire[0] = right
        elif step == 1:
            self.pos_fire[0] = left
        elif step == 2:
            self.pos_fire[1] = down
        elif step == 3:
            self.pos_fire[1] = up

        self.heatmap -= self.rate # Tempture decreasing in this world
        self.heatmap[self.heatmap < self.cov_max] = self.temp_min
        self.heatmap_pad = np.pad(self.heatmap, np.ceil(self.agents[0].r_s))
        self.heatmap[left:right, up:down] = self.cov_max
        
    def get_cost_dist2fire(self):
        return



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
