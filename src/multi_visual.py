import numpy as np
import cv2
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

def simulate(world, ref_states_list, cat_states_list, heatmaps, obstacles, cat_controls_list, num_frames, step_horizon_list, N, init_list, save=False):
    def plot_heatmap(world, obstacle, i):
        '''
        Inputs:
        world.heatmap: 2-D matrix with size (w, h). Heatmap.
        obstacle: Binary 2-D matrix with size (w, h).
        world.agents: A list contains all the agents of Agent type. Each one has 2-D position

        return:
        hm_show: RGB map containing heatmap, obstacles and agents.
        '''
        # Normalize heatmap
        hm_normed = ((world.temp_max - heatmaps[i]) / world.temp_max * 255).astype(np.uint8)  # substitute world.temp_max with 0.1 will be more apparent
        
        # Colormap
        blue_colormap = np.zeros((256, 1, 3), dtype=np.uint8)  # BGR
        blue_colormap[:, 0, 0] =  np.zeros(256)#np.linspace(0, 100, 256)  # Blue channel  0 - 100
        blue_colormap[:, 0, 1] =   np.zeros(256)#np.linspace(0, 100, 256)# Green channel  0 - 255
        blue_colormap[:, 0, 2] = np.linspace(100, 200, 256)  # Red channel  0 - 100
        hm_show = cv2.applyColorMap(hm_normed, blue_colormap)

        return hm_show
    

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

        hm.set_data(np.ones(world.heatmap.shape))
        return path_list, horizon_list

    def animate(i):
        for k in range(n_agents):
            # get variables
            x = cat_states_list[k][0, 0, i]
            y = cat_states_list[k][1, 0, i]
            th = cat_states_list[k][2, 0, i]

            # # update path
            # if i == 0:
            #     path_list[k].set_data(np.array([]), np.array([]))
            # x_new = np.hstack((path_list[k].get_xdata(), x))
            # y_new = np.hstack((path_list[k].get_ydata(), y))
            # path_list[k].set_data(x_new, y_new)

            # update horizon
            x_new = cat_states_list[k][0, 20:, i]
            y_new = cat_states_list[k][1, 20:, i]
            horizon_list[k].set_data(x_new, y_new)

            # update current_state
            current_state_list[k].set_xy(create_triangle([x, y, th], update=True))

        # update heatmap
        img = plot_heatmap(world, None, i)
        hm.set_data(img)

        return path, horizon

    # create figure and axes
    n_agents = len(world.agents)
    fig, ax = plt.subplots(figsize=(6, 6))
    size_world = world.heatmap.shape
    min_scale = 0
    ax.set_xlim(left = min_scale, right = world.len_grid * size_world[0])
    ax.set_ylim(bottom = min_scale, top = world.len_grid * size_world[1])

    # Obstacles
    for (ox, oy, obsr) in obstacles:
        circle = plt.Circle((ox, oy), obsr, color='r')
        ax.add_patch(circle)

    # create lines:
    #   path
    path_list = []
    ref_path_list = []
    horizon_list = []
    current_state_list = []
    for k in range(n_agents):
        path, = ax.plot([], [], 'r', linewidth=2)
        ref_path, = ax.plot([], [], 'b', linewidth=2)
        horizon, = ax.plot([], [], 'x-g', alpha=0.5)
        current_triangle = create_triangle(init_list[k, :])
        current_state = ax.fill(current_triangle[:, 0], current_triangle[:, 1], color='y')
        current_state = current_state[0]

        path_list.append(path)
        ref_path_list.append(ref_path)
        horizon_list.append(horizon)
        current_state_list.append(current_state)


    hm = plt.imshow(np.ones(heatmaps[0].shape)*255, origin='lower', extent=[0., world.len_grid * size_world[0], 0, world.len_grid * size_world[1]])
    plt.xlabel('x position')
    plt.ylabel('y position')
    blue_cmp = plt.get_cmap('seismic', 256)
    blue_cmp = ListedColormap(blue_cmp(np.linspace(0, 0.3, 256)))
    
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=blue_cmp),
             ax=ax, orientation='vertical',fraction=0.046, pad=0.04, label='Percent Coverage')
    
    legend_elements = [Line2D([0], [0], marker='>', color='y', markerfacecolor='y', markersize=15, label='Robots'),
                       Line2D([0], [0], marker='o',color='r', markerfacecolor='r', markersize=15,label='Obstacles',),
                   Line2D([0], [0], marker='x',color='g', markerfacecolor='g', markersize=15,label='MPC Predicted Path',),
                   ]

    ax.legend(handles=legend_elements, loc='upper right')

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
    len_grid = 1
    world = GridWorld(size_world=size_world, len_grid=len_grid, obstacles=None)

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

    # TODO Move the definition of obstacles to env, not in agents
    # init_states = np.array([[0, 0, 0], [30, 20, 0]]) # [x, y, theta]
    n_agents = 3
    n_targets = 3
    # xy_goals = np.random.rand(n_agents, n_targets, 2)
    xy_goals = np.array([[[5, 45], [20, 10], [5, 5]], [[10, 10], [45, 5], [35, 30]], [[45, 30], [15, 45], [30, 35]]], dtype=np.float32) / 50
    # xy_goals = np.array([[[5, 45], [20, 10]], [[10, 10], [45, 5]], [[45, 45], [15, 45]]], dtype=np.float32)
    xy_goals[:, :, 0] = xy_goals[:, :, 0] * size_world[0] * len_grid
    xy_goals[:, :, 1] = xy_goals[:, :, 1] * size_world[1] * len_grid
    theta_goals = np.random.rand(n_agents, n_targets, 1)
    state_goal_list = np.dstack((xy_goals, theta_goals)) # All goal points, state_goal_list[i, 0, :] is the init position for agent i
    # state_goal_list = np.array([[40, 25, np.pi/2], [5, 40, np.pi/2]])
    t0_list = [0 for i in range(n_agents)]
    agents = [MPC_CBF_Unicycle(i, dt, N, v_lim, omega_lim, Q, R, init_state=state_goal_list[i, 0, :], obstacles= obstacles, flag_cbf=True) for i in range(n_agents)]
    ref_states_list = []
    print('Generating ref trajectories...')
    for k in range(n_targets-1):
        for i in range(n_agents):
            path_x, path_y, path_yaw, _, _ = plan_dubins_path(state_goal_list[i, k, 0], state_goal_list[i, k, 1], state_goal_list[i, k, 2],
                                                            state_goal_list[i, k+1, 0], state_goal_list[i, k+1, 1], state_goal_list[i, k+1, 2], r, step_size=v*dt)
            if k == 0:
                ref_states_list.append(np.array([path_x, path_y, path_yaw]).T)
            else:
                ref_states_list[i] = np.concatenate((ref_states_list[i], np.array([path_x, path_y, path_yaw]).T), axis=0)

    world.add_agents(agents)
    state_0_list = [casadi.DM(state_goal_list[i, 0, :]) for i in range(n_agents)]
    u0_list = [casadi.DM.zeros((agents[i].n_controls, N)) for i in range(n_agents)]
    X0_list = [casadi.repmat(state_0_list[i], 1, N + 1) for i in range(n_agents)]
    cat_states_list = [dm_to_array(X0_list[i]) for i in range(n_agents)]
    cat_controls_list = [dm_to_array(u0_list[i][:, 0]) for i in range(n_agents)]

    heatmaps = [np.copy(world.heatmap)]
    trip_lens = [len(ref_states) for ref_states in ref_states_list]
    longest_trip = np.max(trip_lens)
    print('Computing MPC trajectories...')
    for i in range(longest_trip):
        for j in range(n_agents):
            if i < len(ref_states_list[j]):
                u, X_pred = agents[j].solve(X0_list[j], u0_list[j], ref_states_list[j], i)
            
                cat_states_list[j] = np.dstack((cat_states_list[j], dm_to_array(X_pred)))
                cat_controls_list[j] = np.dstack((cat_controls_list[j], dm_to_array(u[:, 0])))
                
                t0_list[j], X0_list[j], u0_list[j] = agents[j].shift_timestep(dt, t0_list[j], X_pred, u)
                agents[j].states = X0_list[j][:, 1]
       
        world.step_heatmap()   
        _ =world.check()     
        heatmaps.append(np.copy(world.heatmap))

    align_length(cat_states_list, longest_trip+1)
    align_length(cat_controls_list, longest_trip+1)
    print('Drawing...')
    simulate(world, ref_states_list, cat_states_list, heatmaps, obstacles, cat_controls_list, longest_trip, dt, N,
         state_goal_list[:, 0, :], save=True)

if __name__ == "__main__":
    main()

