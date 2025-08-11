import pickle
import numpy as np
from env import GridWorld
from visualization import simulate, animate_probs
from datetime import datetime


if __name__=='__main__':
    size_world = (30, 30)
    len_grid = 1
    heatmap = np.ones(size_world) * 0
    heatmap[20:40, 25:45] = 0.8
    world = GridWorld(size_world, len_grid, heatmap, obstacles=None)
    n_agents = 4
    epochs = 20
    date = '2025-08-09'
    affix = f'agent{n_agents}_epoch{epochs}_distributed_ppo_shuffled_clipped' + f'_{date}'
    # affix = f'agent{n_agents}_epoch{epochs}_centralized_ppo_shuffled'
    
    with open(f'./results/traj_log/train_traj_' + affix + '.pkl', 'rb') as f:
        log_dict = pickle.load(f)

    # Downsampling the frames
    interval = 10
    cat_states_list = log_dict['cat_states_list']
    for i in range(len(cat_states_list)):
        cat_states_list[i] = cat_states_list[i][:, :, ::interval]
    heatmaps = log_dict['heatmaps'][::interval]
    cov_lvls =  log_dict['cov_lvls'][::interval]
    probs = log_dict['probs']
    goals = log_dict['xy_goals']
    n_inner = len(heatmaps) / len(goals)
    goals = np.repeat(np.array(goals), n_inner, axis=0)
    probs = np.repeat(np.array(probs), n_inner, axis=0)
    obstacles = log_dict['obstacles']
    n_frames = cat_states_list[0].shape[-1] - 1
    init_states = np.array([cat_states_list[i][:, 0, 0] for i in range(n_agents)])
    simulate(world, n_agents, cat_states_list, heatmaps, cov_lvls, goals, obstacles,
              n_frames, init_states, save=True, save_path='results/visual_train_' + affix + '.mp4')
    animate_probs(probs, size_world, save=True, save_path='results/visual_probs_' + affix + '.mp4')
    print('Animations saved!')