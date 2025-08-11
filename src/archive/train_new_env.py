import os
import pickle
import casadi
import torch
import numpy as np
import hypers
from utils import action2waypoints
from env import GridWorld, GridWorldEnv
import torch.nn.functional as F
from torch.distributions import Categorical
from networks.gcn import GraphConvNet, GCNPos, NetTest, NetCentralized
from mpc_cbf.robot_unicycle import MPC_CBF_Unicycle
from mpc_cbf.plan_dubins import plan_dubins_path
from utils import dm_to_array
from torch.utils.tensorboard import SummaryWriter

def get_theta_goals(xy_goal, pos):
    diff = xy_goal - pos.view(n_agents, 2)
    theta_goal = torch.atan2(diff[:, 1], diff[:, 0]).view(n_agents, 1)
    return theta_goal

def compute_gradient_norm(model, norm_type=2):
    with torch.no_grad():
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1.0 / norm_type)
    return total_norm

def train(world, optim):
    pass


if __name__=='__main__':
    # Set training to be deterministic
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Training settings
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_agents = 4
    epochs = 10
    n_inner = 60
    batch_size = 64
    T = 10
    r_s = 3
    r_c = 50
    hypers.init([5, 5, 0.1])
    size_world = (30, 30)
    len_grid = 1
    # heatmap = np.random.uniform(0.1, 0.5, size_world)
    heatmap = np.ones(size_world)
    for i in range(6):
        heatmap[:, int(i*5):int((i+1)*5)] = 0.2 * i

    world = GridWorldEnv(size_world, len_grid, heatmap, obstacles=None, n_agents=n_agents)
    affix = f'agent{n_agents}_epoch{epochs}_centralized_ppo'
    writer = SummaryWriter(log_dir='runs/overfitting/' + affix)
    thre = np.mean(heatmap * 0.3)
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-4

    # MPC-CBF parameters
    Q_x = 10
    Q_y = 10
    Q_theta = 1
    R_v = 0.5
    R_omega = 0.01
    dt = 0.1
    N = 30
    r = 3 
    v = 5

    v_lim = [0, v]
    omega_lim = [-casadi.pi/3, casadi.pi/3]
    Q = [Q_x, Q_y, Q_theta]
    R = [R_v, R_omega]
    obstacles = [(14,4,0), (18,15,0), (6,19,0), (29, 44,0), (38,15,0), (36,29,0), (25, 26,0)]

    save = False

    # TODO Move the definition of obstacles to env, not in agents. Agents must be able to detect obstacles on the run.
    x_init = np.random.uniform(0., size_world[0], (n_agents, 1))
    y_init = np.random.uniform(0., size_world[1], (n_agents, 1))
    theta_init= np.random.rand(n_agents, 1)
    state_init = np.concat((x_init, y_init, theta_init), axis=-1)
    agents = [MPC_CBF_Unicycle(i, dt, N, v_lim, omega_lim, Q, R, init_state=state_init[i, :], obstacles = obstacles, flag_cbf=True, r_s=r_s, r_c=r_c) for i in range(n_agents)]
    
    decisionNN = NetCentralized(size_world, n_agents).to(dev)
    decisionNN.train()

    # Data for visualization
    cat_states_list = [np.tile(agents[i].states[..., None], (1, N+1)) for i in range(n_agents)]
    heatmaps = []
    cov_lvls = []
    xy_goals_all = []

    # Training
    for ep in range(epochs):
        # optim.zero_grad()
        for t in range(T):
            print(t)
            train(world, optim)

    net_dict = decisionNN.state_dict()
    log_dict = {'cat_states_list': cat_states_list,
                'heatmaps': heatmaps,
                'cov_lvls': cov_lvls,
                'xy_goals': xy_goals_all,
                'obstacles': obstacles}
    
    with open(f'./results/traj_log/train_traj_' + affix + '.pkl', 'wb') as f:
        pickle.dump(log_dict, f)

    if save:
        torch.save({'net_dict': net_dict}, 'results/saved_models/model_' + affix+ '.tar')
        