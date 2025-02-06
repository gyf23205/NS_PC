import numpy as np
import cv2
import torch
import hypers
from utils import action2waypoints
from networks.gcn import GraphConvNet
import casadi
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
from mpc_cbf.plan_dubins import plan_dubins_path
from mpc_cbf.robot_unicycle import MPC_CBF_Unicycle
from utils import dm_to_array, align_length
from env import GridWorld
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

mpl.rcParams['font.size'] = 14
mpl.rcParams['text.usetex'] = True

def simulate(world, ref_states_list, cat_states_list, heatmaps, cov_lvls, obstacles, cat_controls_list, num_frames, step_horizon_list, N, init_list, save=False):
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
        hm_normed = ((world.heat_max - heatmaps[i]) / world.heat_max * 255).astype(np.uint8)  # substitute world.temp_max with 0.1 will be more apparent
        
        # Colormap
        blue_colormap = np.zeros((256, 1, 3), dtype=np.uint8)  # BGR
        blue_colormap[:, 0, 0] =  np.zeros(256) #np.linspace(0, 100, 256)  # Blue channel  0 - 100
        blue_colormap[:, 0, 1] =   np.zeros(256) #np.linspace(0, 100, 256)# Green channel  0 - 255
        blue_colormap[:, 0, 2] = np.linspace(100, 200, 256)  # Red channel  0 - 100
        hm_show = cv2.applyColorMap(hm_normed, blue_colormap)

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
        cl_normed = ((world.cov_max - cov_lvls[i]) / world.cov_max * 255).astype(np.uint8)  # substitute world.temp_max with 0.1 will be more apparent
        
        # Colormap
        red_colormap = np.zeros((256, 1, 3), dtype=np.uint8)  # BGR
        red_colormap[:, 0, 0] =  np.zeros(256) #np.linspace(0, 100, 256)  # Blue channel  0 - 100
        red_colormap[:, 0, 1] = np.linspace(100, 200, 256)  # Red channel  0 - 100
        red_colormap[:, 0, 2] =   np.zeros(256) #np.linspace(0, 100, 256)# Green channel  0 - 255
        cl_show = cv2.applyColorMap(cl_normed, red_colormap)

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

        # hm.set_data(np.ones(world.heatmap.shape))
        return path_list, horizon_list

    def animate(i):
        for k in range(n_agents):
            # get variables
            x = cat_states_list[k][0, 0, i]
            y = cat_states_list[k][1, 0, i]
            th = cat_states_list[k][2, 0, i]

            # update horizon
            x_new = cat_states_list[k][0, 20:, i]
            y_new = cat_states_list[k][1, 20:, i]
            horizon_list[k].set_data(x_new, y_new)

            # update current_state
            current_state_list[k].set_xy(create_triangle([x, y, th], update=True))

        # update heatmap
        img_hm = plot_heatmap(world, i)
        hm.set_data(img_hm)

        img_cl = plot_cov_lvl(world, i)
        cl.set_data(img_cl)

        return path, horizon

    # create figure and axes
    n_agents = len(world.agents)
    fig, ax = plt.subplots(1, 2, figsize=(6, 6))
    size_world = world.heatmap.shape
    min_scale = 0
    ax[0].set_xlim(left = min_scale, right = world.len_grid * size_world[0])
    ax[0].set_ylim(bottom = min_scale, top = world.len_grid * size_world[1])

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
        current_state = ax[0].fill(current_triangle[:, 0], current_triangle[:, 1], color='y')
        current_state = current_state[0]

        path_list.append(path)
        ref_path_list.append(ref_path)
        horizon_list.append(horizon)
        current_state_list.append(current_state)


    hm = ax[0].imshow(np.ones(heatmaps[0].shape)*255, origin='lower', extent=[0., world.len_grid * size_world[0], 0, world.len_grid * size_world[1]])
    cl = ax[1].imshow(np.ones(cov_lvls[0].shape)*255, origin='lower', extent=[0., world.len_grid * size_world[0], 0, world.len_grid * size_world[1]])
    ax[0].set_xlabel('x position')
    ax[1].set_xlabel('y position')
    blue_cmp = plt.get_cmap('seismic', 256)
    blue_cmp = ListedColormap(blue_cmp(np.linspace(0, 0.3, 256)))
    
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=blue_cmp),
             ax=ax[0], orientation='vertical',fraction=0.046, pad=0.04, label='Percent Coverage')
    
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
        sim.save('results/heatmap.mp4', writer='ffmpeg', fps=50)
    plt.show()
    return sim

def main(args=None):
    size_world = (50, 50)
    # rate = np.ones(size_world)*0.01
    len_grid = 1
    heatmap = np.ones(size_world) * 0.1
    heatmap[20:30, 20:30] = 0.6
    world = GridWorld(size_world, len_grid, heatmap, obstacles=None)

    Q_x = 10
    Q_y = 10
    Q_theta = 10
    R_v = 0.5
    R_omega = 0.005

    dt = 0.1
    N = 60

    r = 3 
    v = 1

    v_lim = [-2, 2]
    omega_lim = [-casadi.pi/4, casadi.pi/4]
    Q = [Q_x, Q_y, Q_theta]
    R = [R_v, R_omega]
    obstacles = [(14,4,3), (18,15,3), (6,19,3), (29, 44,3), (38,15,3), (36,29,3), (25, 26,3)]
    n_agents = 3
    theta_goals = np.random.rand(n_agents, n_targets, 1)

    xy_init = np.random.uniform(0., 50, (n_agents, 2))
    theta_init= np.random.rand(n_agents, 1)
    state_init = np.concat((xy_init, theta_init), axis=-1)
    r_s = 10
    agents = [MPC_CBF_Unicycle(i, dt, N, v_lim, omega_lim, Q, R, init_state=state_init[i], obstacles = obstacles, flag_cbf=True, r_s=r_s) for i in range(n_agents)]

    # Load decisionNN model
    decisionNN = GraphConvNet(hypers.n_embed_channel, size_kernal=3, dim_observe=2*r_s, size_world=size_world, n_rel=hypers.n_rel, n_head=4)
    checkpoint = torch.load('results/saved_models/model.tar')
    decisionNN.load_state_dict(checkpoint['net_dict'])
    decisionNN.eval()
    for i in range(n_agents):
        # During the training time, all the agents share the same decision network. Modify this configuration can achieve distributed learning.
        agents[i].decisionNN = decisionNN

    world.add_agents(agents)
    n_targets = 50
    ref_states_list = []
    t0_list = [0 for i in range(n_agents)]
    state_0_list = [casadi.DM(state_init[i]) for i in range(n_agents)]
    u0_list = [casadi.DM.zeros((agents[i].n_controls, N)) for i in range(n_agents)]
    X0_list = [casadi.repmat(state_0_list[i], 1, N + 1) for i in range(n_agents)]
    cat_states_list = [dm_to_array(X0_list[i]) for i in range(n_agents)]
    cat_controls_list = [dm_to_array(u0_list[i][:, 0]) for i in range(n_agents)]
    heatmaps = [np.copy(world.heatmap)]
    cov_lvls = [np.copy(world.cov_lvl)]

    for k in range(n_targets):
        agents = world.agents
        observations = world.check()
        dists = world.agents_dist()
        n_inner = 5
        for i in range(n_agents):
            agents[i].update_neighbors(dists[i, :])
            agents[i].embed_local(torch.tensor(observations[i], dtype=torch.float32))
        
        for i in range(n_agents):
            # Get new waypoints
            neighbor_embed = [agents[j].local_embed for j in agents[i].neighbors]
            
            actions, log_prob = agents[i].generate_waypoints(observations[i], torch.cat(neighbor_embed, dim=0))
            xy_goals = action2waypoints(actions, size_world, len_grid)
            theta_goals = np.random.rand(1) * np.pi - np.pi # Randomly generating theta goal for now.
            state_goal = np.concat((xy_goals, theta_goals), axis=-1)

            # Generating ref trajectory
            path_x, path_y, path_yaw, _, _ = plan_dubins_path(agents[i].states[0], agents[i].states[1], agents[i].states[2],
                                                            state_goal[0], state_goal[1], state_goal[2], r, step_size=v*dt)
            ref_states = np.array([path_x, path_y, path_yaw]).T
            if k == 0:
                ref_states_list.append(ref_states)
            else:
                ref_states_list[i] = np.concatenate((ref_states_list[i], ref_states), axis=0)

            # Generating MPC trajectory
            n_inner = np.min([n_inner, len(ref_states)])

            for j in range(n_inner):
                u, X_pred = agents[i].solve(X0_list[i], u0_list[i], ref_states_list[i], j)
                cat_states_list[i] = np.dstack((cat_states_list[i], dm_to_array(X_pred)))
                cat_controls_list[i] = np.dstack((cat_controls_list[i], dm_to_array(u[:, 0])))
                t0_list[i], X0_list[i], u0_list[i]= agents[i].shift_timestep(dt, t0_list[i], X_pred, u)
        
        world.step()   
        # _ =world.check()     
        heatmaps.append(np.copy(world.heatmap))
        cov_lvls.append(np.copy(world.cov_lvl))

   

if __name__ == "__main__":
    main()

