import pickle
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
    log_probs = []
    ref_states = []

    # Each agent gets local observations
    for i in range(n_agents):
        agents[i].update_neighbors(dists[i, :])
        agents[i].embed_local(torch.tensor(observations[i], dtype=torch.float32))

    for i in range(n_agents):
        # Get new waypoints
        neighbor_embed = [agents[j].local_embed for j in agents[i].neighbors]    
        actions, log_prob = agents[i].generate_waypoints(observations[i], torch.cat(neighbor_embed, dim=0))
        log_probs.append(log_prob)
        xy_goals = action2waypoints(actions, size_world, len_grid)
        theta_goals = np.random.rand(1) * np.pi - np.pi # Randomly generating theta goal for now.
        state_goal = np.concat((xy_goals, theta_goals), axis=-1)

        # Generating ref trajectory
        path_x, path_y, path_yaw, _, _ = plan_dubins_path(agents[i].states[0], agents[i].states[1], agents[i].states[2],
                                                        state_goal[0], state_goal[1], state_goal[2], r, step_size=v*dt)
        ref_states.append(np.array([path_x, path_y, path_yaw]).T)


    cost_agent_list = []
    cost_world_list = []
    for k in range(n_inner): # Multiply MPC steps for each training step
        for i in range(n_agents):
            # Generating MPC trajectory
            u0 = casadi.DM.zeros((agents[i].n_controls, N))
            X0 = casadi.repmat(agents[i].states, 1, N + 1)
            n_inner = np.min([n_inner, len(ref_states[i])])
            t0 = 0
            # cost_agent = []
            u, X_pred = agents[i].solve(X0, u0, ref_states[i], k)
            t0, X0, u0 = agents[i].shift_timestep(dt, t0, X_pred, u)
            agents[i].states = X0[:, 1]

            # Log for visualization
            cat_states_list[i] = np.dstack((cat_states_list[i], dm_to_array(X_pred)))

            # cost_agent.append(torch.tensor(world.get_agent_cost(agents[i].id)))

        world.step()
        cost_world_list.append(torch.tensor(world.get_cost_mean()))
        # cost_world_list.append(torch.tensor(world.get_cost_max()))
        heatmaps.append(np.copy(world.heatmap))
        cov_lvls.append(np.copy(world.cov_lvl))

    cost_agents = torch.zeros((n_agents,), device=dev) # No costs of individual agent now
    cost_world = torch.mean(torch.tensor(cost_world_list).to(dev))
    cost = cost_agents + cost_world
    loss = torch.matmul(torch.tensor(log_probs, device=dev, requires_grad=True), cost)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print("Cost world:%.3f"%cost_world.detach().cpu().numpy())
    print('loss: %.3f'%loss.detach().cpu().numpy())
    
    writer.add_scalar("Cost/world", cost_world.detach().cpu().numpy(), epoch)
    # writer.add_scalar("Cost/agent", np.mean(cost_agent_list), epoch)
    writer.add_scalar("Loss/train", loss.detach().cpu().numpy(), epoch)
    writer.flush()
 


if __name__=='__main__':
    # Set training to be deterministic
    seed = 10
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_agents = 10
    epochs = 1500
    hypers.init([5, 5, 0.1])
    size_world = (50, 50)
    len_grid = 1
    heatmap = np.ones(size_world) * 0
    heatmap[20:35, 20:35] = 0.6 # * np.random.uniform(0, 1, (20, 20))
    world = GridWorld(size_world, len_grid, heatmap, obstacles=None)
    affix = f'agent{n_agents}_epoch{epochs}_mean'
    writer = SummaryWriter(log_dir='runs/overfitting/' + affix)

    Q_x = 10
    Q_y = 10
    Q_theta = 10
    R_v = 0.5
    R_omega = 0.005
    r_s = 3

    dt = 0.1
    N = 20

    r = 3 
    v = 1

    v_lim = [-2, 2]
    omega_lim = [-casadi.pi/4, casadi.pi/4]
    Q = [Q_x, Q_y, Q_theta]
    R = [R_v, R_omega]
    obstacles = [(14,4,3), (18,15,3), (6,19,3), (29, 44,3), (38,15,3), (36,29,3), (25, 26,3)]

    save = True

    # TODO Move the definition of obstacles to env, not in agents. Agents must be able to detect obstacles on the run.
    xy_init = np.random.uniform(0., 50, (n_agents, 2))
    # xy_init[0, :] = np.array([25, 25])
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
        