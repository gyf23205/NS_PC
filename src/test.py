import os
import pickle
import casadi
import torch
import numpy as np
import hypers
from utils import action2waypoints
from env import GridWorld
from networks.gcn import GraphConvNet, GCNPos, NetCentralized
from mpc_cbf.robot_unicycle import MPC_CBF_Unicycle
from mpc_cbf.plan_dubins import plan_dubins_path
from utils import dm_to_array
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def test_voronoi_sup(world):
    criteria = torch.nn.CrossEntropyLoss()
    agents = world.agents
    observations = world.check()
    dists = world.agents_dist()
    eps = 1e-7

    for i in range(n_agents):
        agents[i].update_neighbors(dists[i, :])
        agents[i].embed_local(torch.tensor(observations[i], dtype=torch.float32), size_world, len_grid)

    # Get policy for all agents
    logits_all = []
    ref_states = []
    ub = [size_world[1] * len_grid - 0.1, size_world[0] * len_grid - 0.1, casadi.inf]
    lb = [0, 0, -casadi.inf]
    goals = []
    for i in range(n_agents):
        # Get new waypoints
        neighbor_embed = [agents[j].local_embed for j in agents[i].neighbors]
        actions, log_prob, logits = agents[i].generate_waypoints(observations[i], torch.cat(neighbor_embed, dim=0), size_world, len_grid)
        logits_all.append(logits)
        xy_goals = action2waypoints(actions, size_world, len_grid)
        goals.append(xy_goals)
        dx = xy_goals[0] - agents[i].states[0]
        dy = xy_goals[1] - agents[i].states[1]
        theta_goal = np.degrees(np.atan2(dy, dx))[None, ...]
        
        # theta_goals = np.array([theta])
        state_goal = np.concat((xy_goals, theta_goal), axis=-1)

        # Generating ref trajectory
        path_x, path_y, path_yaw, _, _ = plan_dubins_path(agents[i].states[0], agents[i].states[1], agents[i].states[2],
                                                        state_goal[0], state_goal[1], state_goal[2], r, step_size=v*dt)
        ref_states.append(np.array([path_x, path_y, path_yaw]).T)
    logits_all = torch.stack(logits_all)

    # Get supervisions
    weights = (world.cov_lvl + eps) * world.heatmap
    grid_agent_dist = world.grid_agents_dist()
    clusters = np.argmin(grid_agent_dist, axis=0)

    targets = []
    goals_oracle = []
    for i in range(n_agents):
        mask = clusters == i
        x = np.sum(world.x_coord[mask] * (weights[mask]/np.sum(weights[mask])))
        y = np.sum(world.y_coord[mask] * (weights[mask]/np.sum(weights[mask])))
        goals_oracle.append((x, y))
        x_grid, y_grid = np.floor(2 * x) / 2, np.floor(2 * y) / 2
        idx = int((y_grid - 0.5 * len_grid) * size_world[1] + x // len_grid)
        targets.append(idx)
    targets = torch.tensor(targets, dtype=int, device=dev)
    loss = criteria(logits_all, targets)

    diff = 0
    for i in range(n_agents):
        diff += np.linalg.norm(goals[i] - goals_oracle[i])
    print(diff)
    writer.add_scalar('Diff', diff, ep)
    u0_list = [casadi.DM.zeros((agents[i].n_controls, N)) for i in range(n_agents)]
    X0_list = [casadi.repmat(agents[i].states, 1, N + 1) for i in range(n_agents)]
    t0_list = [0 for i in range(n_agents)]
    cost_world_list = []
    # n_inner = np.min([n_inner, len(ref_states[i])])
    for k in range(n_inner): # Multiply MPC steps for each training step
        for i in range(n_agents):
            # Generating MPC trajectory
            # cost_agent = []
            u, X_pred = agents[i].solve(X0_list[i], u0_list[i], ref_states[i], k, ub, lb)
            t0_list[i], X0_list[i], u0_list[i] = agents[i].shift_timestep(dt, t0_list[i], X_pred, u)
            agents[i].states = X0_list[i][:, 1]

            # Log for visualization
            cat_states_list[i] = np.dstack((cat_states_list[i], dm_to_array(X_pred)))

        world.step()
        cost_world_list.append(torch.tensor(world.get_cost_mean(thre=thre)))
        heatmaps.append(np.copy(world.heatmap))
        cov_lvls.append(np.copy(world.cov_lvl))
    cost_world = torch.sum(torch.tensor(cost_world_list).to(dev))
    writer.add_scalar("Cost/world", cost_world.sum().detach().cpu().numpy(), ep)
    # writer.add_scalar("Cost/agent", cost_agents.detach().cpu().numpy(), ep * T + t)
    writer.add_scalar("Loss/train", loss.detach().cpu().numpy(), ep)
    

if __name__=='__main__':
    # Set training to be deterministic
    seed = 5
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_agents = 4
    epochs_test = 100
    T = 10
    n_inner = 50
    hypers.init([5, 5, 0.1])
    size_world = (30, 30)
    len_grid = 1
    # heatmap = np.random.uniform(0.1, 0.5, size_world)
    heatmap = np.ones(size_world)
    for i in range(6):
        heatmap[:, int(i*5):int((i+1)*5)] = 0.2 * i # * np.random.uniform(0, 1, (20, 20))
    # upper_bound = np.mean(heatmap)
    # upper_bound = torch.tensor(upper_bound, dtype=torch.float32, device=dev)
    # plt.imshow(heatmap)
    # plt.show()
    world = GridWorld(size_world, len_grid, heatmap, obstacles=None)
    affix = f'agent{n_agents}_epoch{epochs_test}_pre_only'
    writer = SummaryWriter(log_dir='runs/test/' + affix)
    thre = np.mean(heatmap * 0.3)
    
    Q_x = 10
    Q_y = 10
    Q_theta = 1
    R_v = 0.5
    R_omega = 0.01
    r_s = 3
    r_c = 20

    dt = 0.1
    N = 30

    r = 3 
    v = 5

    v_lim = [0.0, v]
    omega_lim = [-casadi.pi/4, casadi.pi/4]
    Q = [Q_x, Q_y, Q_theta]
    R = [R_v, R_omega]
    obstacles = [(14,4,0), (18,15,0), (6,19,0), (29, 44,0), (38,15,0), (36,29,0), (25, 26,0)]
    save = True

    # TODO Move the definition of obstacles to env, not in agents. Agents must be able to detect obstacles on the run.
    x_init = np.random.uniform(0., size_world[0], (n_agents, 1))
    y_init = np.random.uniform(0., size_world[1], (n_agents, 1))
    # xy_init[0, :] = np.array([25, 25])
    theta_init= np.random.rand(n_agents, 1)
    state_init = np.concat((x_init, y_init, theta_init), axis=-1)
    # t0_list = [0 for i in range(n_agents)]
    agents = [MPC_CBF_Unicycle(i, dt, N, v_lim, omega_lim, Q, R, init_state=state_init[i], obstacles = obstacles, flag_cbf=True, r_s=r_s, r_c=r_c) for i in range(n_agents)]
    # decisionNN = GraphConvNet(hypers.n_embed_channel, size_kernal=3, dim_observe=2*r_s, size_world=size_world, n_rel=hypers.n_rel, n_head=4).to(dev)
    decisionNN = NetCentralized(hypers.n_embed_channel, size_kernal=3, dim_observe=int(2 * np.ceil(r_s / len_grid) + 1),
                         size_world=size_world, n_rel=hypers.n_rel, n_head=4).to(dev)
    state_dict = torch.load('results/pretrained_models/model_agent4_epoch50.tar', weights_only=True)['net_dict']
    decisionNN.load_state_dict(state_dict)
    # for i in range(n_agents):
    #     # During the training time, all the agents share the same decision network. Modify this configuration can achieve distributed learning.
    #     agents[i].decisionNN = decisionNN
    world.add_agents(agents)
    decisionNN.eval()

    # Data for visualization
    cat_states_list = [np.tile(agents[i].states[..., None], (1, N+1)) for i in range(n_agents)]
    heatmaps = []
    cov_lvls = []

    # Test
    for ep in range(epochs_test):
        test_voronoi_sup(world)

    log_dict = {'cat_states_list': cat_states_list,
                'heatmaps': heatmaps,
                'cov_lvls': cov_lvls,
                'obstacles': obstacles}
    
    with open(f'./results/traj_log/test_traj_' + affix + '.pkl', 'wb') as f:
        pickle.dump(log_dict, f)
        