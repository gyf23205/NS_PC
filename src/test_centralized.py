import os
import pickle
import casadi
import torch
import numpy as np
import hypers
from utils import action2waypoints
from env import GridWorld
import torch.nn.functional as F
from torch.distributions import Categorical
from networks.gcn import GraphConvNet, GCNPos, NetCentralized
from networks.ACnet import ACNetCentralized
from mpc_cbf.robot_unicycle import MPC_CBF_Unicycle
from mpc_cbf.plan_dubins import plan_dubins_path
from utils import dm_to_array
from torch.utils.tensorboard import SummaryWriter

def test(world):
    '''
    Define one training episode.
    For each agent, generate one waypoint, following by several MPC steps.
    '''
    agents = world.agents
    # observations = world.check()
    log_probs = []
    ref_states = []
    ub = [size_world[1] * len_grid - 0.1, size_world[0] * len_grid - 0.1, casadi.inf]
    lb = [0, 0, -casadi.inf]

    # Get new waypoints
    pos = torch.tensor(world.get_agents_pos(), dtype=torch.float32, device=dev).view(1, -1)
    heatmap = torch.tensor(world.heatmap, dtype=torch.float32, device=dev).view(1, 1, size_world[0], size_world[1])
    cov_lvl = torch.tensor(world.cov_lvl, dtype=torch.float32, device=dev).view(1, 1, size_world[0], size_world[1])
    logits, _ = decisionNN(heatmap, cov_lvl, pos)
    probs = F.softmax(torch.squeeze(logits), dim=-1)
    print(torch.argmax(probs, dim=1))
    print(torch.max(probs, dim=1))
    dist = Categorical(probs)
    actions = dist.sample()
    log_prob = dist.log_prob(actions)
    log_probs.append(log_prob)
    xy_goals = action2waypoints(actions, size_world, len_grid)
    # xy_goals = np.array([20., 20.])
    # xy_goals = torch.tensor([[0.5, 29.5], [29.5, 0.5], [29.5, 29.5]], dtype=torch.float32, device=dev)
    xy_goals_all.append(xy_goals.detach().cpu().numpy())
    probs_all.append(probs.detach().cpu().numpy())
    print(xy_goals)
    # theta_goals = get_theta_goals(xy_goals, pos)
    theta_goals = torch.zeros((n_agents, 1), dtype=torch.float32, device=dev)
    state_goal = torch.cat((xy_goals, theta_goals), dim=-1).detach().cpu().numpy()
    # print(state_goal)
    # Generating ref trajectory
    for i in range(n_agents):
        path_x, path_y, path_yaw, _, _ = plan_dubins_path(agents[i].states[0], agents[i].states[1], agents[i].states[2],
                                                        state_goal[i, 0], state_goal[i, 1], state_goal[i, 2], r, step_size=v*dt)
        ref_states.append(np.array([path_x, path_y, path_yaw]).T)


    for k in range(n_mpc): # Multiply MPC steps for each training step
        u0_list = [casadi.DM.zeros((agents[i].n_controls, N)) for i in range(n_agents)]
        X0_list = [casadi.repmat(agents[i].states, 1, N + 1) for i in range(n_agents)]
        t0_list = [0 for i in range(n_agents)]
        # for j in range(n_inner):
        for i in range(n_agents):
            # Generating MPC trajectory
            # cost_agent = []
            # if n_mpc % 5 == 0:
            u, X_pred = agents[i].solve(X0_list[i], u0_list[i], ref_states[i], k, ub, lb)
            t0_list[i], X0_list[i], u0_list[i] = agents[i].shift_timestep(dt, t0_list[i], X_pred, u)
            agents[i].states = X0_list[i][:, 1]


            # Log for visualization
            cat_states_list[i] = np.dstack((cat_states_list[i], dm_to_array(X_pred)))

            # cost_agent.append(torch.tensor(world.get_agent_cost(agents[i].id)))

        world.step()
        heatmaps.append(np.copy(world.heatmap))
        cov_lvls.append(np.copy(world.cov_lvl))

    # cost_world = torch.sum(torch.tensor(cost_world_list).to(dev))
    cov_mean_after = world.get_cost_mean(thre=thre)

    print('cov_mean', cov_mean_after)

    print()

    writer.add_scalar("Cost/mean", cov_mean_after, ep * T + t)
    writer.flush()
 


if __name__=='__main__':
    # Set training to be deterministic
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_agents = 4
    epochs = 20
    epochs_test = 5
    n_mpc = 60
    n_inner = 5
    T = 1
    hypers.init([5, 5, 0.1])
    size_world = (30, 30)
    len_grid = 1
    # heatmap = np.random.uniform(0.1, 0.5, size_world)
    heatmap = np.ones(size_world)
    for i in range(6):
        heatmap[:, int(i*5):int((i+1)*5)] = 0.2 * i # * np.random.uniform(0, 1, (20, 20))

    world = GridWorld(size_world, len_grid, heatmap, obstacles=None)
    affix = f'agent{n_agents}_epoch{epochs}_centralized_ppo_shuffled'
    writer = SummaryWriter(log_dir='runs/overfitting/' + affix)
    thre = np.mean(heatmap * 0.3)
    lr = 1e-4
    # # Define hyperparameters
    # hparams = {
    #     "learning_rate": 0.01,
    #     "batch_size": 64,
    #     "num_epochs": 10,
    # }   


    Q_x = 10
    Q_y = 10
    Q_theta = 1
    R_v = 0.5
    R_omega = 0.01
    r_s = 3
    r_c = 50
    dt = 0.1
    N = 30
    r = 3 
    v = 5

    v_lim = [0, v]
    omega_lim = [-casadi.pi/3, casadi.pi/3]
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
    # state_init = np.array([[15, 15, 3*np.pi/4],[15, 15, -np.pi/4],[15, 15, np.pi/4]])
    # t0_list = [0 for i in range(n_agents)]
    agents = [MPC_CBF_Unicycle(i, dt, N, v_lim, omega_lim, Q, R, init_state=state_init[i, :], obstacles = obstacles, flag_cbf=True, r_s=r_s, r_c=r_c) for i in range(n_agents)]
    # decisionNN = GraphConvNet(hypers.n_embed_channel, size_kernal=3, dim_observe=2*r_s, size_world=size_world, n_rel=hypers.n_rel, n_head=4).to(dev)
    # decisionNN = GCNPos(hypers.n_embed_channel, size_kernal=3, dim_observe=2*r_s, size_world=size_world, n_rel=hypers.n_rel, n_head=4).to(dev)
    decisionNN = ACNetCentralized(size_world, n_agents).to(dev)
    decisionNN.load_state_dict(torch.load('results/saved_models/model_' + affix + '.tar', weights_only=True)['net_dict'])
    
    world.add_agents(agents)
    optim = torch.optim.Adam(decisionNN.parameters(), lr=lr)
    decisionNN.eval()
    # Data for visualization
    cat_states_list = [np.tile(agents[i].states[..., None], (1, N+1)) for i in range(n_agents)]
    heatmaps = []
    cov_lvls = []
    xy_goals_all = []
    probs_all = []

    # Test
    with torch.no_grad():
        for ep in range(epochs_test):
            # optim.zero_grad()
            for t in range(T):
                print(t)
                test(world)

    log_dict = {'cat_states_list': cat_states_list,
                'heatmaps': heatmaps,
                'cov_lvls': cov_lvls,
                'xy_goals': xy_goals_all,
                'probs': probs_all,
                'obstacles': obstacles}
    
    with open(f'./results/traj_log/train_traj_' + affix + '.pkl', 'wb') as f:
        pickle.dump(log_dict, f)

        