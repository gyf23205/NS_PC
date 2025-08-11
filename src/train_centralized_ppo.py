import os
from tqdm import tqdm
import casadi
import torch
import numpy as np
import hypers
from utils import action2waypoints
from env import GridWorld
from networks.ACnet import ACNetCentralized
import torch.nn.functional as F
from torch.distributions import Categorical
from mpc_cbf.robot_unicycle import MPC_CBF_Unicycle
from mpc_cbf.plan_dubins import plan_dubins_path
from utils import dm_to_array
from torch.utils.tensorboard import SummaryWriter
import wandb
from datetime import datetime

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

def collect_trajectory(world, steps):
    '''
    Define one training episode.
    For each agent, generate one waypoint, following by several MPC steps.
    '''
    agents = world.agents
    # observations = world.check()
    # log_probs = []
    ub = [size_world[1] * len_grid - 0.1, size_world[0] * len_grid - 0.1, casadi.inf]
    lb = [0, 0, -casadi.inf]

    trajectory = []
    for _ in tqdm(range(steps), desc="Collecting Trajectory"):
        ref_states = []
        # Get new waypoints
        pos = torch.tensor(world.get_agents_pos(), dtype=torch.float32, device=dev).view(1, -1)
        heatmap = torch.tensor(world.heatmap, dtype=torch.float32, device=dev).view(-1, 1, size_world[0], size_world[1]) # (batch, channel, height, width)
        cov_lvl = torch.tensor(world.cov_lvl, dtype=torch.float32, device=dev).view(-1, 1, size_world[0], size_world[1]) # (batch, channel, height, width)
        obs = (heatmap, cov_lvl, pos)
        with torch.no_grad():
            logits, val = decisionNN(*obs)

        probs = F.softmax(torch.squeeze(logits), dim=-1)
        dist = Categorical(probs)
        actions = dist.sample()
        log_prob = dist.log_prob(actions)
        # log_probs.append(log_prob)
        xy_goals = action2waypoints(actions, size_world, len_grid)
        # theta_goals = get_theta_goals(xy_goals, pos)
        theta_goals = torch.zeros((n_agents, 1), dtype=torch.float32, device=dev)
        state_goal = torch.cat((xy_goals, theta_goals), dim=-1).detach().cpu().numpy()

        for i in range(n_agents):
            path_x, path_y, path_yaw, _, _ = plan_dubins_path(agents[i].states[0], agents[i].states[1], agents[i].states[2],
                                                            state_goal[i, 0], state_goal[i, 1], state_goal[i, 2], r, step_size=v*dt)
            ref_states.append(np.array([path_x, path_y, path_yaw]).T)


        
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
        
        heatmap_next = torch.tensor(world.heatmap, dtype=torch.float32, device=dev)
        cov_lvl_next = torch.tensor(world.cov_lvl, dtype=torch.float32, device=dev)
        pos_next = torch.tensor(world.get_agents_pos(), dtype=torch.float32, device=dev).view(1, -1)
        obs_next = (heatmap_next, cov_lvl_next, pos_next)
        reward_agents = 0
        reward_world = torch.tensor(world.get_reward_mean(thre=thre))
        reward = (reward_agents + reward_world).sum()
        trajectory.append((obs, actions, reward, log_prob, val))
        obs = obs_next
    return trajectory

def process_trajectory(trajectory):
    obs, actions, rewards, log_probs, values= zip(*trajectory)
    heatmaps, cov_lvls, pos = zip(*obs)
    heatmaps = torch.cat(heatmaps, dim=0)  # (steps, n_agents, height, width)
    cov_lvls = torch.cat(cov_lvls, dim=0)  # (steps, n_agents, height, width)
    pos = torch.cat(pos, dim=0)
    actions = torch.stack(actions)
    old_log_probs = torch.stack(log_probs)
    values = torch.stack(values)
    # Compute the advantages
    advantages = [] # Mind the symbol, it is not the same as the cost!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    returns = []
    gae = 0
    next_value = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        next_value = values[t]
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])

    obs = (heatmaps, cov_lvls, pos)
    return obs, actions, old_log_probs, torch.stack(advantages), torch.stack(returns), torch.mean(torch.tensor(rewards))

def ppo_update(model, optimizer, obs, actions, old_log_probs, advantages, returns):
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    heatmaps, cov_lvls, pos = obs
    for _ in range(ppo_epochs):
        # Shuffle the order of minibatch
        idx = torch.randperm(len(actions))
        shuffled_heatmaps = heatmaps[idx]
        shuffled_cov_lvls = cov_lvls[idx]
        shuffled_pos = pos[idx]
        shuffled_actions = actions[idx]
        shuffled_old_log_probs = old_log_probs[idx]
        shuffled_advantages = advantages[idx]
        shuffled_returns = returns[idx]

        for i in range(0, len(actions), mini_batch_size):
            end = i + mini_batch_size
            batch = slice(i, end)

            logits, values = model(shuffled_heatmaps[batch], shuffled_cov_lvls[batch], shuffled_pos[batch])
            dist = Categorical(logits=logits)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(shuffled_actions[batch])
            
            ratio = torch.exp(new_log_probs - shuffled_old_log_probs[batch])
            surr1 = ratio * shuffled_advantages[batch].view(-1, 1)
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * shuffled_advantages[batch].view(-1, 1)

            policy_loss = torch.min(surr1, surr2).mean()
            
            value_loss = (shuffled_returns[batch] - values.squeeze()).pow(2).mean()

            loss = -(policy_loss - c1 * value_loss + c2 * entropy) # Gradient ascent

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decisionNN.parameters(), max_norm=1.0)
            optimizer.step()
            
            wandb.log({
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "gradient_norm": compute_gradient_norm(decisionNN)
            })


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
    n_inner = 30 # 60!!!!!!
    T = 10
    gamma = 0.2
    hypers.init([5, 5, 0.1])
    size_world = (30, 30)
    len_grid = 1
    # heatmap = np.random.uniform(0.1, 0.5, size_world)
    heatmap = np.ones(size_world)
    for i in range(6):
        heatmap[:, int(i*5):int((i+1)*5)] = 0.2 * i # * np.random.uniform(0, 1, (20, 20))

    world = GridWorld(size_world, len_grid, heatmap, obstacles=None)
    date = datetime.now().strftime("%Y-%m-%d")
    affix = f'agent{n_agents}_epoch{epochs}_centralized_ppo_shuffled_clipped_{date}'
    writer = SummaryWriter(log_dir='runs/overfitting/' + affix)

    # Training settings
    thre = np.mean(heatmap * 0.3)
    lr = 1e-4
    gamma = 0.99
    lam = 0.95
    clip_epsilon = 0.2
    lr = 3e-4
    ppo_epochs = 10
    mini_batch_size = 64 # 64
    steps_per_rollout = 256 # !!!
    c1 = 0.6 # 0.5
    c2 = 0.01

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

    # Initialize wandb
    wandb.login(key='1888b9830153065d084181ffc29812cd1011b84b')
    wandb.init(project="NSPC", name="centralized-ppo" + date, config={
        "n_agents": n_agents,
        "epochs": epochs,
        "n_inner": n_inner,
        "T": T,
        "gamma": gamma,
        "lr": lr,
        "ppo_epochs": ppo_epochs,
        "mini_batch_size": mini_batch_size,
        "steps_per_rollout": steps_per_rollout,
        "c1": c1,
        "c2": c2
    })

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
    # for i in range(n_agents):
    #     # During the training time, all the agents share the same decision network. Modify this configuration can achieve distributed learning.
    #     agents[i].decisionNN = decisionNN
    world.add_agents(agents)
    optim = torch.optim.Adam(decisionNN.parameters(), lr=lr)
    decisionNN.train()
    # Data for visualization
    cat_states_list = [np.tile(agents[i].states[..., None], (1, N+1)) for i in range(n_agents)]
    # heatmaps = []
    # cov_lvls = []
    xy_goals_all = []

    best_return = -np.inf

    
    # Training
    for ep in range(epochs):
        trajectory = collect_trajectory(world, steps_per_rollout)
        obs, actions, old_log_probs, advantages, returns, rewards = process_trajectory(trajectory)
        ppo_update(decisionNN, optim, obs, actions, old_log_probs, advantages, returns)

        current_return = np.sum(returns.detach().cpu().numpy()) / steps_per_rollout
        if current_return > best_return:
            best_return = current_return
            if save:
                torch.save({'net_dict': decisionNN.state_dict()}, 'results/saved_models/model_' + affix+ '.tar')
        wandb.log({
            "mean_reward": rewards.detach().cpu().numpy(),
            'return': current_return
        })

        if ep % 3 == 0:
            print(f"Iteration {ep}: reward = {rewards:.2f}")
            print(f"Iteration {ep}: return = {current_return:.2f}")

    # net_dict = decisionNN.state_dict()
    # log_dict = {'cat_states_list': cat_states_list,
    #             'heatmaps': heatmaps,
    #             'cov_lvls': cov_lvls,
    #             'xy_goals': xy_goals_all,
    #             'obstacles': obstacles}
    
    # with open(f'./results/traj_log/train_traj_' + affix + '.pkl', 'wb') as f:
    #     pickle.dump(log_dict, f)

    # if save:
    #     torch.save({'net_dict': net_dict}, 'results/saved_models/model_' + affix+ '.tar')

    wandb.finish()
        