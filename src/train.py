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

def train(world, optim, loss_hist):
    '''
    Define one training step.
    Agent only need to generate new set of waypoints when all the previous generated waypoints have all been visited.
    Decentralized trainig logic but centralized implementation.
    '''
    agents = world.agents
    # Get observations and share embeded information with neighbors
    world.step_heatmap()
    observations = world.check()
    dists = world.get_dist()
    n_inner = 3
    for i in range(n_agents):
        agents[i].update_neighbors(dists[i, :])
        agents[i].embed_local(torch.tensor(observations[i], dtype=torch.float32))
    
    for i in range(n_agents):
        # Get new waypoints
        neighbor_embed = [agents[j].local_embed for j in agents[i].neighbors]
        
        actions, log_prob = agents[i].generate_waypoints(observations[i], torch.cat(neighbor_embed, dim=0))
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
        cost_agent = []
        for k in range(n_inner):
            u, X_pred = agents[i].solve(X0, u0, ref_states, k)
            t0, X0, u0 = agents[i].shift_timestep(dt, t0, X_pred, u)
            # Now all the cost seem to be the same
            cost_agent.append(torch.tensor(world.get_agent_cost(agents[i].id)))
        
        cost_agent = agents[i].get_discount_cost(torch.tensor(cost_agent)).to(dev)

        # Update decisionNN and the world
        # Currently as long as robots don't step on each others' tail, the cost (no matter avg or max) remain the same. Need to figure out a better setting.
        cost_world = torch.tensor(world.get_cost_max()).to(dev)
        cost = (cost_agent + cost_world) - 2 * torch.tensor(world.temp_max).to(dev)
        cost_hist.append((cost_agent.detach().cpu().numpy(), cost_world.detach().cpu().numpy()))
        world.step_heatmap()
        loss = log_prob * cost # Using vanilla REINFORCE for now. May not converge at all.
        loss_hist.append(loss.detach().cpu().numpy())
        optim.zero_grad()
        loss.backward()
        optim.step()
 


if __name__=='__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_agents = 3
    epochs = 10
    hypers.init([5, 5, 0.1])
    size_world = (50, 50)
    len_grid = 1
    rate = np.ones(size_world) * 0.03
    world = GridWorld(size_world=size_world, len_grid=len_grid, obstacles=None, rate=rate)

    Q_x = 10
    Q_y = 10
    Q_theta = 10
    R_v = 0.5
    R_omega = 0.005
    r_s = 10

    dt = 0.1
    N = 60

    r = 3 
    v = 1

    v_lim = [-2, 2]
    omega_lim = [-casadi.pi/4, casadi.pi/4]
    Q = [Q_x, Q_y, Q_theta]
    R = [R_v, R_omega]
    obstacles = [(14,4,3), (18,15,3), (6,19,3), (29, 44,3), (38,15,3), (36,29,3), (25, 26,3)]

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
    loss_hist = []
    cost_hist = []
    for i in range(epochs):
        print(i)
        train(world, optim, loss_hist)
    print(loss_hist)
    print(cost_hist)
    net_dict = decisionNN.state_dict()
    torch.save({'net_dict': net_dict}, 'results/saved_models/model.tar')