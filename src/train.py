import casadi
import torch
import numpy as np
import hypers
from utils import action2waypoints
from env import GridWorld
from gcn import GraphConvNet
from mpc_cbf.robot_unicycle import MPC_CBF_Unicycle

if __name__=='__main__':
    hypers.init([5, 5, 4])
    size_world = (50, 50)
    len_grid = 1
    world = GridWorld(size_world=size_world, len_grid=len_grid, obstacles=None)

    Q_x = 10
    Q_y = 10
    Q_theta = 10
    R_v = 0.5
    R_omega = 0.005
    dim_observe = 10

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
    agents = [MPC_CBF_Unicycle(i, dt, N, v_lim, omega_lim, Q, R, init_state=state_goal_list[i, 0, :], obstacles = obstacles, flag_cbf=True) for i in range(n_agents)]
    ref_states_list = []
    observes = torch.tensor(np.random.normal(size=(n_agents, 1, dim_observe, dim_observe)), dtype=torch.float32)
    neighbor_embed = []
    
    for i in range(n_agents):
        agents[i].set_decision_NN(dim_observe=dim_observe)
        agents[i].embed_local(observes[i, :, :, :])
        neighbor_embed.append(agents[i].local_observe)

    actions = []
    for i in range(n_agents):
        actions.append(agents[i].generate_waypoints(observes[i], torch.cat(neighbor_embed, dim=0)))
    actions = np.array(actions)
    waypoints = action2waypoints(actions, size_world, len_grid)
    print(waypoints.shape)
    