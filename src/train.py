import casadi
import torch
import numpy as np
import hypers
from utils import action2waypoints
from env import GridWorld
from networks.gcn import GraphConvNet
from mpc_cbf.robot_unicycle import MPC_CBF_Unicycle
from mpc_cbf.plan_dubins import plan_dubins_path
from utils import dm_to_array
from torch.utils.tensorboard import SummaryWriter



def train(world, optim):
    '''
    Define one training episode.
    For each agent, generate one waypoint, following by several MPC steps.
    '''
    agents = world.agents
    observations = world.check()
    dists = world.agents_dist()
    n_inner = 5
    cost_world_hist = [] # World cost of the whole episode and all agents
    probs = []
    for ep in range(ep_len):
        # Each agent gets local observations
        for i in range(n_agents):
            agents[i].update_neighbors(dists[i, :])
            agents[i].embed_local(torch.tensor(observations[i], dtype=torch.float32))
        
        # cost_agent_list = []
        for i in range(n_agents):
            # Get new waypoints
            neighbor_embed = [agents[j].local_embed for j in agents[i].neighbors]
            
            actions, log_prob = agents[i].generate_waypoints(observations[i], torch.cat(neighbor_embed, dim=0))
            probs.append(log_prob)
            xy_goals = action2waypoints(actions, size_world, len_grid)
            theta_goals = np.random.rand(1) * np.pi - np.pi # Randomly generating theta goal for now.
            state_goal = np.concat((xy_goals, theta_goals), axis=-1)

            # Generating ref trajectory
            path_x, path_y, path_yaw, _, _ = plan_dubins_path(agents[i].states[0], agents[i].states[1], agents[i].states[2],
                                                            state_goal[0], state_goal[1], state_goal[2], r, step_size=v*dt)
            ref_states = np.array([path_x, path_y, path_yaw]).T

            # Generating MPC trajectory
            u0 = casadi.DM.zeros((agents[i].n_controls, N))
            X0 = casadi.repmat(agents[i].states, 1, N + 1)
            n_inner = np.min([n_inner, len(ref_states)])
            t0 = 0
            # cost_agent = []
            for k in range(n_inner):
                u, X_pred = agents[i].solve(X0, u0, ref_states, k)
                t0, X0, u0 = agents[i].shift_timestep(dt, t0, X_pred, u)
                # Now all the cost seem to be the same
                # cost_agent.append(torch.tensor(world.get_agent_cost(agents[i].id)))
            
            # cost_agent = agents[i].get_discount_cost(torch.tensor(cost_agent)).to(dev)

            # Update decisionNN and the world
            # Currently as long as robots don't step on each others' tail, the world cost (no matter avg or max) remain the same.
            cost_world = torch.tensor(world.get_cost_max()).to(dev)
            # cost = (cost_agent + cost_world) - 2 * torch.tensor(world.cov_max).to(dev)
            # cost_hist.append((cost_agent.detach().cpu().numpy(), ))
            # cost_agent_list.append(cost_agent.detach().cpu().numpy())
            cost_world_hist.append(cost_world.detach())
            
            world.step()
    cost_world_hist = torch.tensor(cost_world_hist, device=dev).view(ep_len, n_agents)
    probs = torch.tensor(probs, requires_grad=True, device=dev). view(ep_len, n_agents)
    loss = torch.zeros([], device=dev)
    for ep in range(ep_len):
        row_idx = torch.arange(ep_len - ep, device=dev).view(-1, 1)
        discounts = hypers.discount ** row_idx.expand(-1, n_agents)
        loss += torch.mean(probs[ep, :] * torch.sum(cost_world_hist[ep:, :] * discounts, dim=0))
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    writer.add_scalar("Cost/world", np.mean(cost_world_hist.cpu().numpy()), epoch)
    # writer.add_scalar("Cost/agent", np.mean(cost_agent_list), epoch)
    writer.add_scalar("Loss/train", loss.detach().cpu().numpy(), epoch)
    writer.flush()
 


if __name__=='__main__':
    # Set training to be deterministic
    seed = 4
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter()
    n_agents = 10
    epochs = 200
    ep_len = 10 # Make a continuous task episodic
    hypers.init([5, 5, 0.1])
    size_world = (50, 50)
    len_grid = 1
    heatmap = np.ones(size_world) * 0.1
    heatmap[20:40, 20:40] = 0.6 * np.random.uniform(0, 1, (20, 20))
    world = GridWorld(size_world, len_grid, heatmap, obstacles=None)

    Q_x = 10
    Q_y = 10
    Q_theta = 10
    R_v = 0.5
    R_omega = 0.005
    r_s = 5

    dt = 0.1
    N = 60

    r = 3 
    v = 1

    v_lim = [-2, 2]
    omega_lim = [-casadi.pi/4, casadi.pi/4]
    Q = [Q_x, Q_y, Q_theta]
    R = [R_v, R_omega]
    obstacles = [(14,4,3), (18,15,3), (6,19,3), (29, 44,3), (38,15,3), (36,29,3), (25, 26,3)]

    save = False

    # TODO Move the definition of obstacles to env, not in agents. Agents must be able to detect obstacles on the run.
    xy_init = np.random.uniform(0., 50, (n_agents, 2))
    theta_init= np.random.rand(n_agents, 1)
    state_init = np.concat((xy_init, theta_init), axis=-1)
    # t0_list = [0 for i in range(n_agents)]
    agents = [MPC_CBF_Unicycle(i, dt, N, v_lim, omega_lim, Q, R, init_state=state_init[i], obstacles = obstacles, flag_cbf=True, r_s=r_s) for i in range(n_agents)]
    decisionNN = GraphConvNet(hypers.n_embed_channel, size_kernal=3, dim_observe=2*r_s, size_world=size_world, n_rel=hypers.n_rel, n_head=4).to(dev)
    for i in range(n_agents):
        # During the training time, all the agents share the same decision network. Modify this configuration can achieve distributed learning.
        agents[i].decisionNN = decisionNN
    world.add_agents(agents)
    optim = torch.optim.Adam(decisionNN.parameters())

    for epoch in range(epochs):
        print(epoch)
        train(world, optim)

    net_dict = decisionNN.state_dict()
    if save:
        torch.save({'net_dict': net_dict}, 'results/saved_models/model.tar')