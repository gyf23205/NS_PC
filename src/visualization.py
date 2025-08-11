import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
# from mpc_cbf.plan_dubins import plan_dubins_path
# from mpc_cbf.robot_unicycle import MPC_CBF_Unicycle
# from utils import dm_to_array, align_length
# from env import GridWorld
from matplotlib.lines import Line2D
import pickle

mpl.rcParams['font.size'] = 14
mpl.rcParams['text.usetex'] = True

def simulate(world, n_agents, cat_states_list, heatmaps, cov_lvls, goals, obstacles, num_frames, init_list, save, save_path):
    def plot_heatmap(world, i):
        '''
        Inputs:
        world.heatmap: 2-D matrix with size (w, h). Heatmap.
        obstacle: Binary 2-D matrix with size (w, h).
        world.agents: A list contains all the agents of Agent type. Each one has 2-D position

        return:
        hm_show: RGB map containing heatmap, obstacles and agents.
        '''
        # Normalize heatmap
        hm_normed = (heatmaps[i] / world.heat_max * 255).astype(np.uint8)  # substitute world.temp_max with 0.1 will be more apparent
        
        # # Colormap
        # blue_colormap = np.zeros((256, 1, 3), dtype=np.uint8)  # BGR
        # blue_colormap[:, 0, 0] =  np.zeros(256) #np.linspace(0, 100, 256)  # Blue channel  0 - 100
        # blue_colormap[:, 0, 1] =   np.zeros(256) #np.linspace(0, 100, 256)# Green channel  0 - 255
        # blue_colormap[:, 0, 2] = np.linspace(100, 200, 256)  # Red channel  0 - 100
        # hm_show = cv2.applyColorMap(hm_normed, blue_colormap)
        hm_show = hm_normed
        return hm_show
    
    def plot_cov_lvl(world, i):
        '''
        Inputs:
        world.heatmap: 2-D matrix with size (w, h). Heatmap.
        obstacle: Binary 2-D matrix with size (w, h).
        world.agents: A list contains all the agents of Agent type. Each one has 2-D position

        return:
        hm_show: RGB map containing heatmap, obstacles and agents.
        '''
        # Normalize heatmap
        cl_normed = (cov_lvls[i] / world.cov_max * 255).astype(np.uint8)  # substitute world.temp_max with 0.1 will be more apparent
        
        # Colormap
        # green_colormap = np.zeros((256, 1, 3), dtype=np.uint8)  # BGR
        # green_colormap[:, 0, 0] =  np.zeros(256) #np.linspace(0, 100, 256)  # Blue channel  0 - 100
        # green_colormap[:, 0, 1] = np.linspace(100, 200, 256)  # Red channel  0 - 100
        # green_colormap[:, 0, 2] =   np.zeros(256) #np.linspace(0, 100, 256)# Green channel  0 - 255
        # cl_show = cv2.applyColorMap(cl_normed, green_colormap)
        cl_show = cl_normed
        return cl_show
    

    def create_triangle(state=[0,0,0], h=2, w=1.5, update=False):
        x, y, th = state
        triangle = np.array([
            [h, 0   ],
            [0,  w/2],
            [0, -w/2],
            [h, 0   ]
        ]).T
        rotation_matrix = np.array([
            [np.cos(th), -np.sin(th)],
            [np.sin(th),  np.cos(th)]
        ])

        coords = np.array([[x, y]]) + (rotation_matrix @ triangle).T
        if update == True:
            return coords
        else:
            return coords[:3, :]

    def init():
        scat.set_offsets(np.empty((0, 2)))  
        # hm.set_data(np.ones(world.heatmap.shape))
        return path_list, horizon_list, scat

    def animate(i):
        for k in range(n_agents):
            # get variables
            x = cat_states_list[k][0, 0, i]
            y = cat_states_list[k][1, 0, i]
            th = cat_states_list[k][2, 0, i]

            # update horizon
            x_new = cat_states_list[k][0, :, i]
            y_new = cat_states_list[k][1, :, i]
            horizon_list[k].set_data(x_new, y_new)

            # update current_state
            current_state_list[k].set_xy(create_triangle([x, y, th], update=True))

        # goals_plt.set_data(goals[i][:, 0], goals[i][:, 1])
        scat.set_offsets(goals[i])
        scat.set_color(colors)
        # update heatmap
        img_hm = plot_heatmap(world, i)
        hm.set_data(img_hm)

        img_cl = plot_cov_lvl(world, i)
        cl.set_data(img_cl)

        return horizon_list

    # create figure and axes
    fig, ax = plt.subplots(1, 2, figsize=(6, 6))
    fig.set_size_inches(19.2, 10.8)
    size_world = world.heatmap.shape
    min_scale = 0
    ax[0].set_xlim(left = min_scale, right = world.len_grid * size_world[1])
    ax[0].xaxis.tick_top()
    ax[0].xaxis.set_label_position('top')
    ax[0].set_ylim(bottom = world.len_grid * size_world[0], top = min_scale)
    ax[1].set_xlim(left = min_scale, right = world.len_grid * size_world[1])
    ax[1].xaxis.tick_top()
    ax[1].xaxis.set_label_position('top')
    ax[1].set_ylim(bottom = world.len_grid * size_world[0], top = min_scale)

    colors = ['red', 'purple', 'blue', 'orange']
    scat = ax[0].scatter([], [], s=50, c='r')
    scat.set_color(colors)
    # Obstacles
    for (ox, oy, obsr) in obstacles:
        circle = plt.Circle((ox, oy), obsr, color='r')
        ax[0].add_patch(circle)

        circle1 = plt.Circle((ox, oy), obsr, color='r')
        ax[1].add_patch(circle1)

    # create lines:
    #   path
    path_list = []
    ref_path_list = []
    horizon_list = []
    current_state_list = []
    for k in range(n_agents):
        path, = ax[0].plot([], [], 'r', linewidth=2)
        ref_path, = ax[0].plot([], [], 'b', linewidth=2)
        horizon, = ax[0].plot([], [], 'x-g', alpha=0.5)
        current_triangle = create_triangle(init_list[k, :])
        current_state = ax[0].fill(current_triangle[:, 0], current_triangle[:, 1], color=colors[k])
        current_state = current_state[0]

        path_list.append(path)
        ref_path_list.append(ref_path)
        horizon_list.append(horizon)
        current_state_list.append(current_state)

    # goals_plt = ax[0].scatter(goals[0][:, 0], goals[0][:, 1])
    hm = ax[0].imshow(np.ones(heatmaps[0].shape)*255, origin='lower', extent=[0., world.len_grid * size_world[0], 0, world.len_grid * size_world[1]], cmap='viridis', vmin=0, vmax=255)
    cl = ax[1].imshow(np.ones(cov_lvls[0].shape)*255, origin='lower', extent=[0., world.len_grid * size_world[0], 0, world.len_grid * size_world[1]], cmap='viridis', vmin=0, vmax=255)
    ax[0].set_xlabel('x position')
    ax[1].set_xlabel('y position')
    blue_cmp = plt.get_cmap('viridis', 256)
    cmp = plt.get_cmap('viridis', 256)
    # blue_cmp = ListedColormap(blue_cmp(np.linspace(0, 0.3, 256)))
    # cmp = ListedColormap(cmp(np.linspace(0, 0.3, 256)))
    
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=blue_cmp),
             ax=ax[0], orientation='vertical',fraction=0.046, pad=0.04, label='Importance density')
    
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmp),
             ax=ax[1], orientation='vertical',fraction=0.046, pad=0.04, label='Coverage level')
    
    legend_elements = [Line2D([0], [0], marker='>', color='y', markerfacecolor='y', markersize=15, label='Robots'),
                       Line2D([0], [0], marker='o',color='r', markerfacecolor='r', markersize=15,label='Obstacles',),
                   Line2D([0], [0], marker='x',color='g', markerfacecolor='g', markersize=15,label='MPC Predicted Path',),
                   ]

    ax[0].legend(handles=legend_elements, loc='upper right')

    sim = animation.FuncAnimation(
        fig=fig,
        func = animate,
        init_func=init,
        frames=num_frames,
        interval=0.1,
        blit=False,
        repeat=False
    )
    if save == True:
        sim.save(save_path, writer='ffmpeg', fps=50)
    # plt.show()
    return sim


def plot_cov(cov_lvls, heatmaps):
    cov_low_imp = []
    cov_high_imp = []
    weighted_cov = []
    for i in range(len(cov_lvls)):
        cov_low_imp.append(np.mean(cov_lvls[i][:, 0:15]))
        cov_high_imp.append(np.mean(cov_lvls[i][:, 15:]))
        weighted_cov.append(np.mean(heatmaps[i]*cov_lvls[i]))
        if i > 0:
            print(np.mean(heatmaps[i] - heatmaps[i-1]))

    plt.plot(cov_low_imp)
    plt.plot(cov_high_imp)
    plt.plot(weighted_cov)
    plt.title('Coverage Level')
    plt.legend(['low importance', 'high importance', 'weighted mean'])
    plt.show()
    
def animate_probs(probs, size_world, save=False, save_path='probs_anim.mp4'):
    """
    Animate the probabilities of each agent over time.
    probs: list of np arrays, each of shape (n_agent, size_world[0]*size_world[1])
    size_world: tuple, (height, width)
    """
    # import matplotlib.pyplot as plt
    # from matplotlib import animation

    n_frames = len(probs)
    n_agent = probs[0].shape[0]
    h, w = size_world

    fig, axes = plt.subplots(1, n_agent, figsize=(5*n_agent, 5))
    if n_agent == 1:
        axes = [axes]

    ims = []
    for agent_id in range(n_agent):
        axes[agent_id].set_title(f'Agent {agent_id}')
        axes[agent_id].set_xlabel('X')
        axes[agent_id].set_ylabel('Y')

    def init():
        return []

    def animate(i):
        im_list = []
        for agent_id in range(n_agent):
            axes[agent_id].cla()
            axes[agent_id].set_title(f'Agent {agent_id}')
            axes[agent_id].set_xlabel('X')
            axes[agent_id].set_ylabel('Y')
            prob_map = probs[i, agent_id, :].reshape(h, w)
            # if agent_id == 0:
            #     print(prob_map[:10, :10])  # Print the first 10x10 block of the first agent's probability map
            im = axes[agent_id].imshow(prob_map, origin='lower', cmap='viridis', vmin=np.min(probs), vmax=np.max(probs))
            # fig.colorbar(im, ax=axes[agent_id], fraction=0.046, pad=0.04)
            im_list.append(im)
        return im_list

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=n_frames, interval=200, blit=False, repeat=False
    )
   
    if save:
        anim.save(save_path, writer='ffmpeg', fps=5)
    # plt.show()
    return anim

if __name__ == '__main__':
    size_world = (30, 30)
    len_grid = 1
    probs = []
    for j in range(100):
        prob = np.ones((4, size_world[0] * size_world[1])) * (j / 100)
        probs.append(prob.copy())

    animate_probs(np.stack(probs), size_world, save=True, save_path='probs_anim.mp4')
