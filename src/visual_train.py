import pickle
import numpy as np
from env import GridWorld
from multi_visual import simulate


if __name__=='__main__':
    size_world = (50, 50)
    len_grid = 1
    heatmap = np.ones(size_world) * 0.1
    heatmap[20:40, 25:45] = 0.6
    world = GridWorld(size_world, len_grid, heatmap, obstacles=None)
    n_agents = 10
    epochs = 300
    affix = f'agent{n_agents}_epoch{epochs}_corner'
    
    
    with open(f'./results/traj_log/train_traj_' + affix + '.pkl', 'rb') as f:
        log_dict = pickle.load(f)

    # Downsampling the frames
    interval = 15
    cat_states_list = log_dict['cat_states_list']
    for i in range(len(cat_states_list)):
        cat_states_list[i] = cat_states_list[i][:, :, ::interval]
    heatmaps = log_dict['heatmaps'][::interval]
    cov_lvls =  log_dict['cov_lvls'][::interval]
    obstacles = log_dict['obstacles']
    n_frames = cat_states_list[0].shape[-1] - 1
    init_states = np.array([cat_states_list[i][:, 0, 0] for i in range(n_agents)])
    simulate(world, n_agents, cat_states_list, heatmaps, cov_lvls, obstacles,
              n_frames, init_states, save=True, save_path='results/visual_train.mp4')