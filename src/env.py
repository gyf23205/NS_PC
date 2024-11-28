import numpy as np
import matplotlib.pyplot as plt


class GridWorld(object):
    def __init__(self, size_world, len_grid, rate, obstacles) -> None:
        '''
        The left-up is (0, 0). Down and right are positive directions for y and x.
        size: 2-D integer array. Size of the grid world.
        len_grid: Edge length of each square grid.
        rate: 2-D array with the same size of heatmap, indicating the changing rate of each grid.
        '''
        self.size_world = size_world
        self.obstacle = None # Add later
        self.heatmap = np.random.uniform(0, 1, size_world)
        self.heatmap_pad = None
        self.rate = rate
        self.agents = []
        self.env_state = None
        self.temp_max = 1
        self.len_grid = len_grid
        self.agent_rs = None
        self.x_coord = 0.5 * len_grid + np.repeat(np.arange(size_world[1])[None, :] * len_grid, size_world[0], axis=0)
        self.y_coord = 0.5 * len_grid + np.repeat(np.arange(size_world[0])[:, None] * len_grid, size_world[1], axis=1)

        
    def get_dist(self):
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
        for agent in self.agents:
            # dist = np.sqrt((self.x_coord - agent.states[0])**2 + (self.y_coord - agent.states[1])**2)
            current_grid = (int(agent.states[0] // self.len_grid + self.agent_rs), int(agent.states[1] // self.len_grid + self.agent_rs))
            left, right, up, down = current_grid[0] - self.agent_rs, current_grid[0] + self.agent_rs, current_grid[1] - self.agent_rs, current_grid[1] + self.agent_rs
            # By doing the following, all the agents get sparse observation where only the frontier of their observations has nonzero values.
            # This could be not sufficient amount of information.
            observations.append(self.heatmap_pad[None, left:right, up:down])
        return observations
    
    def get_agent_cost(self, id):
        size = self.heatmap.shape
        agent = None
        for a in self.agents:
            if a.id == id:
                agent = a
                break
        current_grid = (int(agent.states[0] // self.len_grid), int(agent.states[1] // self.len_grid))
        left = np.max([current_grid[0] - self.agent_rs, 0])
        right = np.min([current_grid[0] + self.agent_rs, size[1]])
        up = np.max([current_grid[1] - self.agent_rs, 0])
        down = np.min([current_grid[1] + self.agent_rs, size[1]])
        return -np.mean(self.heatmap[left:right, up:down])

    def step_heatmap(self):
        self.heatmap += self.rate
        self.heatmap[self.heatmap > self.temp_max] = self.temp_max
        self.heatmap_pad = np.pad(self.heatmap, self.agents[0].r_s)
        

    
    def add_agents(self, agents):
        self.agents = agents
        self.agent_rs = self.agents[0].r_s
        self.heatmap_pad = np.pad(self.heatmap, self.agent_rs)

    def get_cost_max(self):
        '''
        Get the maximum value of the heatmap, it could be strict and the cost is always max
        '''
        return np.max(self.heatmap)
    
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
        row, col = np.indices(self.heatmap.shape)  # row indices = column indices^T, 10*10. 
        cost = self.heatmap * dists[idx_agent_cloest, row, col]  # dim(dists[idx_agent_cloest]) = 10*10*10*10, dim(dists[idx_agent_cloest, row, col]) = 10*10
        

        return cost
    

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
        self.heatmap[self.heatmap < self.temp_max] = self.temp_min
        self.heatmap_pad = np.pad(self.heatmap, self.agents[0].r_s)
        self.heatmap[left:right, up:down] = self.temp_max
        
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
