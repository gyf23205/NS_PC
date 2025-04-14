import os
import math
import pickle
import casadi
import torch
import numpy as np
import hypers
from utils import action2waypoints
from env import GridWorld
from networks.gcn import GraphConvNet, GCNPos, NetTest, GCNPosOnly, NetTest1
from mpc_cbf.robot_unicycle import MPC_CBF_Unicycle
from mpc_cbf.plan_dubins import plan_dubins_path
from utils import dm_to_array
from torch.utils.tensorboard import SummaryWriter

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
    observations = world.check()
    dists = world.agents_dist()
    log_probs = []
    costs = []
    # ref_states = []
    # ub = [size_world[1] * len_grid - 0.1, size_world[0] * len_grid - 0.1, casadi.inf]
    # lb = [0, 0, -casadi.inf]
    # lb = [-casadi.inf, -casadi.inf, -casadi.inf]
    # ub = [casadi.inf, casadi.inf, casadi.inf]

    # Each agent gets local observations
    for i in range(n_agents):
        agents[i].update_neighbors(dists[i, :])
        agents[i].embed_local(torch.tensor(observations[i], dtype=torch.float32), size_world, len_grid)

    # ref_states_list = []
    one_explore_steps_xytheta_goals = np.zeros([n_agents, 3])
    for i in range(n_agents):
        # Get new waypoints
        neighbor_embed = [agents[j].local_embed for j in agents[i].neighbors]    
        actions, log_prob, probs = agents[i].generate_waypoints(observations[i], torch.cat(neighbor_embed, dim=0), size_world, len_grid)
        log_probs.append(log_prob)
        xy_goals = action2waypoints(actions, size_world, len_grid)
        # xy_goals = np.array([20., 20.])
        # print(xy_goals)
        dx = xy_goals[0] - agents[i].states[0]
        dy = xy_goals[1] - agents[i].states[1]
        theta = math.degrees(math.atan2(dy, dx))
        
        theta_goals = np.array([theta])
        # theta_goals =  np.array([np.pi/2])#np.random.rand(1) * np.pi - np.pi # Randomly generating theta goal for now.
        state_goal = np.concat((xy_goals, theta_goals), axis=-1)
        one_explore_steps_xytheta_goals[i] = state_goal

        # # Generating ref trajectory
        # path_x, path_y, path_yaw, _, _ = plan_dubins_path(agents[i].states[0], agents[i].states[1], agents[i].states[2],
        #                                                 state_goal[0], state_goal[1], state_goal[2], r, step_size=v*dt)
        # ref_states.append(np.array([path_x, path_y, path_yaw]).T)
        # cat_states_list[i] = np.dstack((cat_states_list[i], state_goal.reshape(-1, 1)))


    cost_agent_list = []
    cost_world_list = []

    for i in range(n_agents):
        agents[i].states = one_explore_steps_xytheta_goals[i]
        cost_agent_list.append(world.get_agent_cost(agents[i].id))
        # Log for visualization
        cat_states_list[i] = np.dstack((cat_states_list[i], one_explore_steps_xytheta_goals[i].reshape(-1, 1)))

            # cost_agent.append(torch.tensor(world.get_agent_cost(agents[i].id)))
    world.step()
    cost_world_list.append(torch.tensor(world.get_cost_mean(thre=thre)))
    # cost_world_list.append(torch.tensor(world.get_cost_max()))
    heatmaps.append(np.copy(world.heatmap))
    cov_lvls.append(np.copy(world.cov_lvl))

    cost_agents = torch.tensor(cost_agent_list, device=dev)
    cost_agents = 0
    cost_world = torch.sum(torch.tensor(cost_world_list).to(dev))
    # cost_world = 0
    cost = (cost_agents + cost_world).sum()
    # loss = torch.matmul(torch.stack(log_probs), cost)
    loss = torch.stack(log_probs).sum() * cost
    # optim.zero_grad() 
    loss.backward()
    optim.step()
    grad_norm = compute_gradient_norm(decisionNN)

    print("Cost:%.3f"%cost.detach().cpu().numpy())
    print('loss: %.3f'%loss.detach().cpu().numpy())
    print('Gradient norm: %.3f'%grad_norm)
    print()
    
    writer.add_scalar("Cost/cost", cost.detach().cpu().numpy(), epoch)
    # writer.add_scalar("Cost/agent", np.mean(cost_agent_list), epoch)
    writer.add_scalar("Loss/train", loss.detach().cpu().numpy(), epoch)
    writer.add_scalar('Norm/grad', grad_norm)
    writer.flush()
 


if __name__=='__main__':
    # Set training to be deterministic
    seed = 5 # 5
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_agents = 4
    epochs = 500
    n_inner = 20
    hypers.init([5, 5, 0.1])
    size_world = (30, 30)
    len_grid = 2
    # heatmap = np.random.uniform(0.1, 0.5, size_world)
    heatmap = np.ones(size_world) * 0.1
    heatmap[10:15, 10:15] = 0.8 # * np.random.uniform(0, 1, (20, 20))
    heatmap[24:29, 0:29] = 0
    # upper_bound = np.mean(heatmap)
    # upper_bound = torch.tensor(upper_bound, dtype=torch.float32, device=dev)
    world = GridWorld(size_world, len_grid, heatmap, obstacles=None)
    affix = f'agent{n_agents}_epoch{epochs}_rand'
    writer = SummaryWriter(log_dir='runs/overfitting/' + affix)
    thre = np.mean(heatmap * 0.3)
    lr = 1e-4
    # Define hyperparameters
    hparams = {
        "n_inner": n_inner,
        "threshold": thre,
        "num_epochs": epochs
    }   

    writer.add_hparams(hparams, {})


    Q_x = 10
    Q_y = 10
    Q_theta = 1
    R_v = 0.5
    R_omega = 0.01
    r_s = 3
    r_c = 0

    dt = 0.1
    N = 0

    r = 3 
    v = 1

    v_lim = [0, 5]
    omega_lim = [-casadi.pi/4, casadi.pi/4]
    Q = [Q_x, Q_y, Q_theta]
    R = [R_v, R_omega]
    obstacles = [(14,4,0), (18,15,0), (6,19,0), (29, 44,0), (38,15,0), (36,29,0), (25, 26,0)]

    save = True

    # TODO Move the definition of obstacles to env, not in agents. Agents must be able to detect obstacles on the run.
    x_init = np.random.uniform(0., size_world[0], (n_agents, 1))
    y_init = np.random.uniform(size_world[1]-1, size_world[1], (n_agents, 1))
    
    # y_init = np.random.uniform(0., size_world[1], (n_agents, 1))
    # xy_init[0, :] = np.array([25, 25])
    theta_init= np.random.rand(n_agents, 1)
    state_init = np.concat((x_init, y_init, theta_init), axis=-1)
    # t0_list = [0 for i in range(n_agents)]
    agents = [MPC_CBF_Unicycle(i, dt, N, v_lim, omega_lim, Q, R, init_state=state_init[i], obstacles = obstacles, flag_cbf=True, r_s=r_s, r_c=r_c) for i in range(n_agents)]
    # decisionNN = GraphConvNet(hypers.n_embed_channel, size_kernal=3, dim_observe=2*r_s, size_world=size_world, n_rel=hypers.n_rel, n_head=4).to(dev)
    decisionNN = NetTest(hypers.n_embed_channel, size_kernal=3, dim_observe=2*r_s, size_world=size_world, n_rel=hypers.n_rel, n_head=4).to(dev)
    for i in range(n_agents):
        # During the training time, all the agents share the same decision network. Modify this configuration can achieve distributed learning.
        agents[i].decisionNN = decisionNN
    world.add_agents(agents)
    optim = torch.optim.Adam(decisionNN.parameters(), lr=lr, weight_decay=1)
    decisionNN.train()
    # Data for visualization
    cat_states_list = [np.tile(agents[i].states[..., None], (1, N+1)) for i in range(n_agents)]
    heatmaps = []
    cov_lvls = []

    # Training
    for epoch in range(epochs):
        print(epoch)
        train(world, optim)

    net_dict = decisionNN.state_dict()
    log_dict = {'cat_states_list': cat_states_list,
                'heatmaps': heatmaps,
                'cov_lvls': cov_lvls,
                'obstacles': obstacles}
    
    with open(f'./results/traj_log/train_traj_' + affix + '.pkl', 'wb') as f:
        pickle.dump(log_dict, f)

    if save:
        torch.save({'net_dict': net_dict}, 'results/saved_models/model_' + affix+ '.tar')
        