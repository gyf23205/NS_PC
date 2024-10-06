import cv2
import numpy as np
from env import GridWorld 
import matplotlib.pyplot as plt


def plot_heatmap(world, obstacle):
    '''
    Inputs:
    world.heatmap: 2-D matrix with size (w, h). Heatmap.
    obstacle: Binary 2-D matrix with size (w, h).
    world.agents: A list contains all the agents of Agent type. Each one has 2-D position

    return:
    hm_show: RGB map containing heatmap, obstacles and agents.
    '''
    # Normalize heatmap
    hm_normed = ((world.temp_max - world.heatmap) / world.temp_max * 255).astype(np.uint8)  # substitute world.temp_max with 0.1 will be more apparent
    
    # Colormap
    green_colormap = np.zeros((256, 1, 3), dtype=np.uint8)  
    green_colormap[:, 0, 0] = np.linspace(0, 100, 256)  # Blue channel  0 - 100
    green_colormap[:, 0, 1] = np.arange(256)  # Green channel  0 - 256
    green_colormap[:, 0, 2] = np.linspace(0, 100, 256)  # Red channel  0 - 100

    hm_show = cv2.applyColorMap(hm_normed, green_colormap)

    # Mark agents
    for agent in world.agents:
        dist = np.sqrt((world.x_coord - agent.state[0])**2 + (world.y_coord - agent.state[1])**2)
        agent_area = dist < agent.r_s
        hm_show[agent_area] = [255, 255, 255]  # White color for agent
    
    # Mark Obstacles
    hm_show[obstacle == 1] = [255, 50, 50]  

    return hm_show


if __name__ == "__main__":
    # initial agent, environment and obstacles
    world = GridWorld((10, 10), 1)

    class Agent(object):
        def __init__(self, init_state) -> None:
            self.state = init_state
            self.r_s = 2.0
    agents = [Agent(np.array([0., 0.]))]
    world.add_agents(agents)

    obstacle = np.zeros((10, 10), dtype=np.uint8)
    obstacle[7:9, 1:3] = 1 
    
    # plot
    for i in range(10):
        world.update_heatmap()
        img = plot_heatmap(world, obstacle)
        world.agents[0].state += 1.0   
        
        plt.imshow(img)
        plt.savefig('./plot_show{}.jpg'.format(i))



    
