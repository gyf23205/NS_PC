import os
import pickle
import casadi
import torch
import numpy as np
import hypers
from scipy.spatial import distance_matrix
from utils import action2waypoints
from env import GridWorld
from networks.gcn import GraphConvNet, GCNPos, NetTest
from mpc_cbf.robot_unicycle import MPC_CBF_Unicycle
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def pretrain(world, optim, pos, targets):
    '''
    pos: (batch, n_agents, 2)
    target: (batch, n_agents)
    '''
    criteria = torch.nn.CrossEntropyLoss()
    agents = world.agents 
    batch_size = pos.shape[0]
    criteria = torch.nn.CrossEntropyLoss()

    loss = 0.0
    # Generate neighbors
    for k in range(batch_size):
        world.init_agents_pos(pos[k, :, :].cpu().numpy())
        dists = world.agents_dist()
        observations = world.check()
        logits_all = []
        for i in range(n_agents):
            agents[i].update_neighbors(dists[i, :])
            agents[i].embed_local(torch.tensor(observations[i], dtype=torch.float32), size_world, len_grid)

        goals = []
        for i in range(n_agents):
            neighbor_embed = [agents[j].local_embed for j in agents[i].neighbors]    
            actions, log_prob, logits = agents[i].generate_waypoints(observations[i], torch.cat(neighbor_embed, dim=0), size_world, len_grid)
            logits_all.append(logits)
            xy_goals = action2waypoints(actions, size_world, len_grid)
            goals.append(xy_goals)
        logits_all = torch.stack(logits_all)

        loss = loss + criteria(logits_all, targets[i, :])
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(loss.item())
    writer.add_scalar("Loss/train", loss.detach().cpu().numpy(), ep * n_batch + idx)
    # print(compute_gradient_norm(decisionNN))
    # diff = 0
    # for i in range(n_agents):
    #     diff += np.linalg.norm(goals[i] - goals_oracle[i])
    # print(diff)
    # writer.add_scalar('Diff', diff, ep)
    # world.step()

def compute_gradient_norm(model, norm_type=2):
    with torch.no_grad():
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


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
    epochs = 0
    epochs_pretrain = 20
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
    affix = f'agent{n_agents}_epoch{epochs_pretrain}'
    writer = SummaryWriter(log_dir='runs/pretrain/' + affix)
    # thre = np.mean(heatmap * 0.3)
    lr = 1e-4

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
    # Define hyperparameters
    hparams = {
        'epochs': epochs,
        'T': T,
        "n_inner": n_inner,
        # 'cost': 'agent+mean_world',
        'r_c':r_c,
        'r_s':r_s,
        'MPC_N': N,
    }   
    # writer.add_hparams(hparams, {})
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
    decisionNN = GCNPos(hypers.n_embed_channel, size_kernal=3, dim_observe=int(2 * np.ceil(r_s / len_grid) + 1),
                         size_world=size_world, n_rel=hypers.n_rel, n_head=4).to(dev)
    for i in range(n_agents):
        # During the training time, all the agents share the same decision network. Modify this configuration can achieve distributed learning.
        agents[i].decisionNN = decisionNN
    world.add_agents(agents)
    optim = torch.optim.Adam(decisionNN.parameters(), lr=lr)
    torch.nn.utils.clip_grad_norm_(decisionNN.parameters(), max_norm=2.)
    decisionNN.train()
    # Data for visualization
    cat_states_list = [np.tile(agents[i].states[..., None], (1, N+1)) for i in range(n_agents)]
    heatmaps = []
    cov_lvls = []

    pos_train = torch.tensor(np.load('data/pretrain/pos_train.npy'), device=dev)
    targets_train = torch.tensor(np.load('data/pretrain/targets_train.npy'), device=dev)
    training_set = TensorDataset(pos_train, targets_train)
    batch_size = 64
    trainloader = DataLoader(training_set, batch_size=batch_size, shuffle=False)
    
    n_batch = len(trainloader)
    # Pretrain
    for ep in range(epochs_pretrain):
        for idx, (pos, targets) in enumerate(trainloader):
            pretrain(world, optim, pos, targets)

    net_dict = decisionNN.state_dict()
    log_dict = {'cat_states_list': cat_states_list,
                'heatmaps': heatmaps,
                'cov_lvls': cov_lvls,
                'obstacles': obstacles}
    
    with open(f'./results/traj_log/train_traj_' + affix + '.pkl', 'wb') as f:
        pickle.dump(log_dict, f)

    if save:
        torch.save({'net_dict': net_dict}, 'results/pretrained_models/model_' + affix+ '.tar')
        