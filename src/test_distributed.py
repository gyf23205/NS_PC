import os
from tqdm import tqdm
import casadi
import torch
import numpy as np
import hypers
from utils import action2waypoints
from env import GridWorld
from networks.ACnet import ACNetCentralized, ACNetDistributed
import torch.nn.functional as F
from torch.distributions import Categorical
from mpc_cbf.robot_unicycle_local_map import MPC_CBF_Map_Unicycle
from mpc_cbf.plan_dubins import plan_dubins_path
from utils import dm_to_array, normalize_pos_tensor
from torch.utils.tensorboard import SummaryWriter
import wandb
from datetime import datetime
import pickle

def get_theta_goals(xy_goal, pos):
    diff = xy_goal - pos.view(n_agents, 2)
    theta_goal = torch.atan2(diff[:, 1], diff[:, 0]).view(n_agents, 1)
    return theta_goal


def test(world):
    '''
    Define one training episode.
    For each agent, generate one waypoint, following by several MPC steps.
    '''
    agents = world.agents
    dists = world.agents_dist()
    ub = [size_world[1] * len_grid - 0.1, size_world[0] * len_grid - 0.1, casadi.inf]
    lb = [0, 0, -casadi.inf]

    ref_states = []
    log_prob_list = []
    # Update neighbors
    for i in range(n_agents):
        agents[i].update_neighbors(dists[i, :])

    # Comunicate local heatmap and cov_lvl with neighbors
    for i in range(n_agents):
        agents[i].share_local_maps(agents)

    # Update heatmap and cov_lvl with received messages from neighbors
    for i in range(n_agents):
        agents[i].update_heatmap_cov()

    # Get pos
    pos = torch.tensor(world.get_agents_pos(), dtype=torch.float32, device=dev)

    # Record observations
    heatmap_temp = []
    cov_lvl_temp = []
    for i in range(n_agents):
        heatmap_temp.append(agents[i].heatmap)
        cov_lvl_temp.append(agents[i].cov_lvl)
    heatmap_all  = torch.tensor(np.array(heatmap_temp), dtype=torch.float32, device=dev).view(1, n_agents, size_world[0], size_world[1])
    cov_lvl_all = torch.tensor(np.array(cov_lvl_temp), dtype=torch.float32, device=dev).view(1, n_agents, size_world[0], size_world[1])
    obs = (heatmap_all, cov_lvl_all, normalize_pos_tensor(pos, size_world, len_grid).view(1, n_agents, 3))
    xy_goals_temp = []
    actions_list = []
    val_list = []

    probs_temp = []
    for i in range(n_agents):
        logits, val = agents[i].generate_waypoints()
        probs = F.softmax(torch.squeeze(logits), dim=-1)
        probs_temp.append(probs.detach().cpu().numpy())
        dist = Categorical(probs)
        actions = dist.sample()
        actions_list.append(actions)
        val_list.append(val)
        log_prob = dist.log_prob(actions)
        log_prob_list.append(log_prob)
        xy_goal = action2waypoints(actions, size_world, len_grid)
        xy_goal = xy_goal.detach().cpu().numpy()
        xy_goals_temp.append(xy_goal)
        dx = xy_goal[0] - agents[i].states[0]
        dy = xy_goal[1] - agents[i].states[1]
        theta_goal = np.degrees(np.atan2(dy, dx))[None, ...]
        
        # theta_goals = np.array([theta])
        state_goal = np.concatenate((xy_goal, theta_goal), axis=-1)
        path_x, path_y, path_yaw, _, _ = plan_dubins_path(agents[i].states[0], agents[i].states[1], agents[i].states[2],
                                                        state_goal[0], state_goal[1], state_goal[2], r, step_size=v*dt)
        ref_states.append(np.array([path_x, path_y, path_yaw]).T)
    xy_goals_all.append(np.stack(xy_goals_temp, axis=0))
    probs_all.append(np.stack(probs_temp, axis=0))

    # trajectory.append((pos.detach().cpu().numpy(), xy_goals.detach().cpu().numpy(), ref_states))
    cost_agent_list = []
    # cost_world_list = []
    
    # n_inner = np.min([n_inner, len(ref_states[i])])
    
    for k in range(n_inner): # Multiply MPC steps for each training step
        u0_list = [casadi.DM.zeros((agents[i].n_controls, N)) for i in range(n_agents)]
        X0_list = [casadi.repmat(agents[i].states, 1, N + 1) for i in range(n_agents)]
        t0_list = [0 for i in range(n_agents)]
        for i in range(n_agents):
            # Generating MPC trajectory
            # cost_agent = []
            # if n_inner % 5 == 0:
            u, X_pred = agents[i].solve(X0_list[i], u0_list[i], ref_states[i], k, ub, lb)
            t0_list[i], X0_list[i], u0_list[i] = agents[i].shift_timestep(dt, t0_list[i], X_pred, u)
            agents[i].states = X0_list[i][:, 1]
            cost_agent_list.append(world.get_agent_cost(agents[i].id))

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

    writer.add_scalar("Cost/mean", cov_mean_after, ep)
    writer.flush()


if __name__=='__main__':
    # Set training to be deterministic
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Exvironment settings
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_agents = 4
    epochs = 20 # 10!!!!
    epochs_test = 50
    n_inner = 40 # 60!!!!!!
    T = 10
    gamma = 0.2
    hypers.init([5, 5, 0.1])
    size_world = (30, 30)
    len_grid = 1
    # heatmap = np.random.uniform(0.1, 0.5, size_world)
    heatmap = np.ones(size_world)
    for i in range(6):
        heatmap[:, int(i*5):int((i+1)*5)] = 0.2 * i # * np.random.uniform(0, 1, (20, 20))

    date = '2025-08-09'
    world = GridWorld(size_world, len_grid, heatmap, obstacles=None)
    affix = f'agent{n_agents}_epoch{epochs}_distributed_ppo_shuffled_clipped' + f'_{date}'
    writer = SummaryWriter(log_dir='runs/overfitting/' + affix)

    thre = np.mean(heatmap * 0.3)


    # MPC settings
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

    # # Initialize wandb
    # wandb.login(key='1888b9830153065d084181ffc29812cd1011b84b')
    # wandb.init(project="NSPC", name="distributed-ppo" + date, config={
    #     "n_agents": n_agents,
    #     "epochs": epochs,
    #     "n_inner": n_inner,
    #     "T": T,
    #     "gamma": gamma,
    #     "lr": lr,
    #     "ppo_epochs": ppo_epochs,
    #     "mini_batch_size": mini_batch_size,
    #     "steps_per_rollout": steps_per_rollout,
    #     "c1": c1,
    #     "c2": c2
    # })

    # TODO Move the definition of obstacles to env, not in agents. Agents must be able to detect obstacles on the run.
    x_init = np.random.uniform(0., size_world[0], (n_agents, 1))
    y_init = np.random.uniform(0., size_world[1], (n_agents, 1))
    # xy_init[0, :] = np.array([25, 25])
    theta_init= np.random.rand(n_agents, 1)
    state_init = np.concat((x_init, y_init, theta_init), axis=-1)
    # state_init = np.array([[15, 15, 3*np.pi/4],[15, 15, -np.pi/4],[15, 15, np.pi/4]])
    # t0_list = [0 for i in range(n_agents)]
    agents = [MPC_CBF_Map_Unicycle(i, dt, N, v_lim, omega_lim, Q, R, init_state=state_init[i, :], world_size=size_world, obstacles=obstacles, flag_cbf=True, r_s=r_s, r_c=r_c) for i in range(n_agents)]
    # decisionNN = GraphConvNet(hypers.n_embed_channel, size_kernal=3, dim_observe=2*r_s, size_world=size_world, n_rel=hypers.n_rel, n_head=4).to(dev)
    # decisionNN = GCNPos(hypers.n_embed_channel, size_kernal=3, dim_observe=2*r_s, size_world=size_world, n_rel=hypers.n_rel, n_head=4).to(dev)
    decisionNN = ACNetDistributed(size_world, n_agents).to(dev)
    decisionNN.load_state_dict(torch.load('results/saved_models/model_' + affix + '.tar', weights_only=True)['net_dict'])
    for i in range(n_agents):
        # During the training time, all the agents share the same decision network. Modify this configuration can achieve distributed learning.
        agents[i].decisionNN = decisionNN
    world.add_agents(agents)
    decisionNN.eval()
    # Data for visualization
    cat_states_list = [np.tile(agents[i].states[..., None], (1, N+1)) for i in range(n_agents)]
    # heatmaps = []
    # cov_lvls = []
    xy_goals_all = []

    # Data for visualization
    cat_states_list = [np.tile(agents[i].states[..., None], (1, N+1)) for i in range(n_agents)]
    heatmaps = []
    cov_lvls = []
    xy_goals_all = []
    probs_all = []

    # Test
    for ep in range(epochs_test):
        print(f"Epoch {ep+1}/{epochs_test}")
        test(world)


    log_dict = {'cat_states_list': cat_states_list,
                'heatmaps': heatmaps,
                'cov_lvls': cov_lvls,
                'xy_goals': xy_goals_all,
                'probs': probs_all,
                'obstacles': obstacles}
    
    with open(f'./results/traj_log/train_traj_' + affix + '.pkl', 'wb') as f:
        pickle.dump(log_dict, f)
        