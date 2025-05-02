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
from networks.gcn import GraphConvNet, GCNPos, NetTest, NetCentralized
from mpc_cbf.robot_unicycle import MPC_CBF_Unicycle
# from mpc_cbf.plan_dubins import plan_dubins_path
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
    '''
    Define one training episode.
    For each agent, generate one waypoint, following by several MPC steps.
    '''
    agents = world.agents
    # observations = world.check()
    log_probs = []
    ub = [size_world[1] * len_grid - 0.1, size_world[0] * len_grid - 0.1, casadi.inf]
    lb = [0, 0, -casadi.inf]

    # Get new waypoints
    pos = torch.tensor(world.get_agents_pos(), dtype=torch.float32, device=dev).view(1, -1)
    heatmap = torch.tensor(world.heatmap, dtype=torch.float32, device=dev)
    cov_lvl = torch.tensor(world.cov_lvl, dtype=torch.float32, device=dev)
    logits = decisionNN(heatmap, cov_lvl, pos)
    probs = F.softmax(logits, dim=-1)
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
    print(xy_goals)
    # theta_goals = get_theta_goals(xy_goals, pos)
    theta_goals = torch.zeros((n_agents, 1), dtype=torch.float32, device=dev)
    state_goal = torch.cat((xy_goals, theta_goals), dim=-1).detach().cpu().numpy()

    cost_agent_list = []
    cost_world_list = []
    
    # n_inner = np.min([n_inner, len(ref_states[i])])
    
    for k in range(n_inner): # Multiply MPC steps for each training step
        u0_list = [casadi.DM.zeros((agents[i].n_controls, N)) for i in range(n_agents)]
        X0_list = [casadi.repmat(agents[i].states, 1, N + 1) for i in range(n_agents)]
        t0_list = [0 for i in range(n_agents)]
        for i in range(n_agents):
            # Generating MPC trajectory
            # cost_agent = []
            # if n_inner % 5 == 0:
            u, X_pred = agents[i].solve(X0_list[i], u0_list[i], state_goal[i, :].reshape(1, -1), k, ub, lb)
            t0_list[i], X0_list[i], u0_list[i] = agents[i].shift_timestep(dt, t0_list[i], X_pred, u)
            agents[i].states = X0_list[i][:, 1]
            cost_agent_list.append(world.get_agent_cost(agents[i].id))

            # Log for visualization
            cat_states_list[i] = np.dstack((cat_states_list[i], dm_to_array(X_pred)))

            # cost_agent.append(torch.tensor(world.get_agent_cost(agents[i].id)))

        world.step()
        # cost_world_list.append(torch.tensor(world.get_cost_mean(thre=thre)))
        # cost_world_list.append(torch.tensor(world.get_cost_max()))
        heatmaps.append(np.copy(world.heatmap))
        cov_lvls.append(np.copy(world.cov_lvl))

    # cost_agents = torch.tensor(cost_agent_list, device=dev)
    cost_agents = 0
    # cost_world = torch.sum(torch.tensor(cost_world_list).to(dev))
    cost_world = torch.tensor(world.get_cost_mean(thre=thre))
    # cost_world = 0
    cost = (cost_agents + cost_world).sum()
    # loss = torch.matmul(torch.stack(log_probs), cost)
    entrophy = (-probs * torch.log(probs)).sum(-1).mean()
    # loss = torch.stack(log_probs).sum() * cost - 0.1 * entrophy
    loss = - entrophy
    optim.zero_grad()
    loss.backward()
    optim.step()
    grad_norm = compute_gradient_norm(decisionNN)

    print("Cost:%.3f"%cost.detach().cpu().numpy())
    print('loss: %.3f'%loss.detach().cpu().numpy())
    print('Gradient norm: %.3f'%grad_norm)
    print()
    
    writer.add_scalar("Cost/cost", cost.detach().cpu().numpy(), ep * T + t)
    # writer.add_scalar("Cost/agent", np.mean(cost_agent_list), epoch)
    writer.add_scalar("Loss/all", loss.detach().cpu().numpy(), ep * T + t)
    writer.add_scalar("Loss/entrophy", entrophy.detach().cpu().numpy(), ep * T + t)
    writer.add_scalar('Norm/grad', grad_norm, ep * T + t)
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
    n_agents = 1
    epochs = 1
    n_inner = 30
    T = 30
    hypers.init([5, 5, 0.1])
    size_world = (30, 30)
    len_grid = 1
    # heatmap = np.random.uniform(0.1, 0.5, size_world)
    heatmap = np.ones(size_world)
    for i in range(6):
        heatmap[:, int(i*5):int((i+1)*5)] = 0.2 * i # * np.random.uniform(0, 1, (20, 20))

    world = GridWorld(size_world, len_grid, heatmap, obstacles=None)
    affix = f'agent{n_agents}_epoch{epochs}_centralized'
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
    Q_theta = 0
    R_v = 0
    R_omega = 0
    r_s = 3
    r_c = 50
    dt = 0.2
    N = 50
    r = 3 
    v = 5

    v_lim = [0, v]
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
    # state_init = np.array([[15, 15, 3*np.pi/4],[15, 15, -np.pi/4],[15, 15, np.pi/4]])
    # t0_list = [0 for i in range(n_agents)]
    agents = [MPC_CBF_Unicycle(i, dt, N, v_lim, omega_lim, Q, R, init_state=state_init[i, :], obstacles = obstacles, flag_cbf=True, r_s=r_s, r_c=r_c) for i in range(n_agents)]
    # decisionNN = GraphConvNet(hypers.n_embed_channel, size_kernal=3, dim_observe=2*r_s, size_world=size_world, n_rel=hypers.n_rel, n_head=4).to(dev)
    # decisionNN = GCNPos(hypers.n_embed_channel, size_kernal=3, dim_observe=2*r_s, size_world=size_world, n_rel=hypers.n_rel, n_head=4).to(dev)
    decisionNN = NetCentralized(size_world, n_agents).to(dev)
    # for i in range(n_agents):
    #     # During the training time, all the agents share the same decision network. Modify this configuration can achieve distributed learning.
    #     agents[i].decisionNN = decisionNN
    world.add_agents(agents)
    optim = torch.optim.Adam(decisionNN.parameters(), lr=lr)
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

    # if save:
    #     torch.save({'net_dict': net_dict}, 'results/saved_models/model_' + affix+ '.tar')
        