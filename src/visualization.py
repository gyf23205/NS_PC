import numpy as np
import cv2
import casadi
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib import animation
from mpc_cbf.plan_dubins import plan_dubins_path
from mpc_cbf.robot_unicycle import MPC_CBF_Unicycle
from utils import dm_to_array
from env import GridWorld



def simulate(world, ref_states, cat_states, heatmaps, cat_controls, num_frames, step_horizon, N, reference, save=False):
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
        green_colormap = np.zeros((256, 1, 3), dtype=np.uint8)  # BGR
        green_colormap[:, 0, 0] = np.linspace(0, 100, 256)  # Blue channel  0 - 100
        green_colormap[:, 0, 1] = np.arange(256)  # Green channel  0 - 255
        green_colormap[:, 0, 2] = np.linspace(0, 100, 256)  # Red channel  0 - 100
        hm_show = cv2.applyColorMap(hm_normed, green_colormap)
        
        # # Mark agents
        # for agent in world.agents:
        #     dist = np.sqrt((world.x_coord - agent.states[0])**2 + (world.y_coord - agent.states[1])**2)
        #     agent_area = dist < agent.r_s
        #     hm_show[agent_area] = [255, 255, 255]  # White color for agent
        # hm_show[int(agent.state[0]), int(agent.state[1]), :] = 0

        # # Mark Obstacles
        # hm_show[obstacle == 1] = [50, 50, 255]  

        return hm_show
    

    def create_triangle(state=[0,0,0], h=1, w=0.5, update=False):
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
        return path, horizon#, current_state, target_state,

    def animate(i):
        # get variables
        x = cat_states[0, 0, i]
        y = cat_states[1, 0, i]
        th = cat_states[2, 0, i]

        # get ref variables
        x_ref = ref_states[:, 0]
        y_ref = ref_states[:, 1]


        # update ref path
        ref_path.set_data(x_ref, y_ref)

        # update path
        if i == 0:
            path.set_data(np.array([]), np.array([]))
        x_new = np.hstack((path.get_xdata(), x))
        y_new = np.hstack((path.get_ydata(), y))
        path.set_data(x_new, y_new)

        # update horizon
        x_new = cat_states[0, :, i]
        y_new = cat_states[1, :, i]
        horizon.set_data(x_new, y_new)

        # update current_state
        current_state.set_xy(create_triangle([x, y, th], update=True))

        # update heatmap
        img = plot_heatmap(world, None, i)
        hm.set_data(img)
        # # update target_state
        # xy = target_state.get_xy()
        # target_state.set_xy(xy)

        return path, horizon#, current_state, target_state,

    # create figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))
    size_world = world.heatmap.shape
    min_scale = 0
    ax.set_xlim(left = min_scale, right = world.len_grid * size_world[0])
    ax.set_ylim(bottom = min_scale, top = world.len_grid * size_world[1])

    # circle = plt.Circle((obs_x, obs_y), obs_diam/2, color='r')
    # ax.add_patch(circle)

    # create lines:
    #   path
    path, = ax.plot([], [], 'r', linewidth=2)

    ref_path, = ax.plot([], [], 'b', linewidth=2)
    #   horizon
    horizon, = ax.plot([], [], 'x-g', alpha=0.5)


    hm = plt.imshow(np.ones(heatmaps[0].shape)*255, origin='lower', extent=[0., world.len_grid * size_world[0], 0, world.len_grid * size_world[1]])

    #   current_state
    current_triangle = create_triangle(reference[:3])
    current_state = ax.fill(current_triangle[:, 0], current_triangle[:, 1], color='r')
    current_state = current_state[0]
    # #   target_state
    # target_triangle = create_triangle(reference[3:])
    # target_state = ax.fill(target_triangle[:, 0], target_triangle[:, 1], color='b')
    # target_state = target_state[0]

    sim = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=num_frames,
        #interval=step_horizon*100,
        interval=100,
        blit=False,
        repeat=False
    )

    if save == True:
        sim.save('results/heatmap.gif', writer='ffmpeg', fps=30)
    plt.show()
    return sim

def main(args=None):
    world = GridWorld(size_world=(20, 20), len_grid=1, obstacles=None)

    Q_x = 10
    Q_y = 10
    Q_theta = 10
    R_v = 0.5
    R_omega = 0.005

    dt = 0.1
    N = 20
    # idx = 0
    t0 = 0

    # x in [0, size_world[0]], y in [0, size_world[1] * world.len_grid]
    x_0 = 0
    y_0 = 0
    theta_0 = 0

    # x in [0, size_world[0]], y in [0, size_world[1]* world.len_grid]
    x_goal = 12
    y_goal = 18
    theta_goal = np.pi/2

    r = 1 
    v = 1
    path_x, path_y, path_yaw, _, _ = plan_dubins_path(x_0, y_0, theta_0, x_goal, y_goal, theta_goal, r, step_size=v*dt)

    ref_states = np.array([path_x, path_y, path_yaw]).T

    v_lim = [-1, 1]
    omega_lim = [-casadi.pi/4, casadi.pi/4]
    Q = [Q_x, Q_y, Q_theta]
    R = [R_v, R_omega]
    obs_list = [(4,0), (8,5), (6,9), (2, -4), (8,-5), (6,-9), (5, -6)]

    # TODO Move the definition of obstacles to env, not in agents
    init_state = [x_0, y_0, theta_0]
    mpc_cbf = MPC_CBF_Unicycle(0, dt,N, v_lim, omega_lim, Q, R, init_state=np.array(init_state), obstacles= obs_list, flag_cbf=True)
    world.add_agents([mpc_cbf])
    state_0 = casadi.DM(init_state)
    u0 = casadi.DM.zeros((mpc_cbf.n_controls, N))
    X0 = casadi.repmat(state_0, 1, N + 1)
    cat_states = dm_to_array(X0)
    cat_controls = dm_to_array(u0[:, 0])

    # x_arr = [x_0]
    # y_arr = [y_0]
    # states_hist = [np.array(init_state)]
    heatmaps = [np.copy(world.heatmap)]
    for i in range(len(ref_states)): 
        u, X_pred = mpc_cbf.solve(X0, u0, ref_states, i)
        
        cat_states = np.dstack((cat_states, dm_to_array(X_pred)))
        cat_controls = np.dstack((cat_controls, dm_to_array(u[:, 0])))
        
        t0, X0, u0 = mpc_cbf.shift_timestep(dt, t0, X_pred, u)
        mpc_cbf.states = X0[:, 1]
        # states_hist.append(mpc_cbf.states)
        world.update_heatmap()
        
        heatmaps.append(np.copy(world.heatmap))
        # x_arr.append(X0[0,1])
        # y_arr.append(X0[1,1])
        # idx += 1
    
    num_frames = len(ref_states)
    # To be replaced
    simulate(world, ref_states, cat_states, heatmaps, cat_controls, num_frames, dt, N,
         np.array([x_0, y_0, theta_0, x_goal, y_goal, theta_goal]), save=False)

if __name__ == "__main__":
    main()



    
