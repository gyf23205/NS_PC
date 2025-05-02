import numpy as np
import matplotlib.pyplot as plt


if __name__=='__main__':
    n_samples = 1000
    n_agents = 4
    size_world = (30, 30)
    len_grid = 1
    eps = 1e-7
    heatmap = np.ones(size_world)
    # world = GridWorld(size_world, len_grid, heatmap, obstacles=None)
    for i in range(6):
        heatmap[:, int(i*5):int((i+1)*5)] = 0.2 * i # * np.random.uniform(0, 1, (20, 20))
    heatmap += eps

    pos = np.random.uniform(0, size_world[0]*len_grid, (n_samples, n_agents, 2))
    # plt.scatter(pos[0, :, 0], pos[0, :, 1], marker='*')
    x_coord = 0.5 * len_grid + np.repeat(np.arange(size_world[1])[None, :] * len_grid, size_world[0], axis=0)
    y_coord = 0.5 * len_grid + np.repeat(np.arange(size_world[0])[:, None] * len_grid, size_world[1], axis=1)
    
    coords = np.stack([x_coord, y_coord], axis=-1) # (30, 30, 2)

    delta = pos[:, :, None, None, :] - coords[None, None, :, :, :] # (n_samples, n_agents, 30, 30)
    dist = np.sum(delta**2, axis=-1)
    # print(dist.shape)
    clusters = np.argmin(dist, axis=1) # (n_sample, 30, 30)
    # plt.imshow(clusters[0, :, :])
    # print(clusters.shape)
    # mask = [] # [(n_sample, 30, 30)]
    # for i in range(n_agents):
    #     mask.append(clusters == i)
    # # print(mask[0].shape)
    # mask = np.stack(mask, axis=1) # (n_samples, n_agents, 30, 30)
    # print(mask.shape)
    mask = np.transpose(clusters[..., None] == np.arange(n_agents), (0, 3, 1, 2))
    coords_exp = coords[None, None, ...]
    sum_xy_weighted = (mask[..., None] * coords_exp * heatmap[None, None, ..., None]).sum(axis=(2, 3)) # (n_sample, n_agents)
    counts_weighted = (mask * heatmap[None, None, ...]).sum(axis=(2, 3))
    target_pos = np.divide(sum_xy_weighted, counts_weighted[..., None], where=counts_weighted[..., None] != 0) # (n_sample, n_agents, 2)
    # plt.scatter(target_pos[0, :, 0], target_pos[0, :, 1])
    # Convert pos to indices
    coords_flat = coords.reshape(size_world[0]*size_world[1], 2)
    d2 = ((coords_flat[None, None, ...] - target_pos[..., None, :]) ** 2).sum(-1)
    # print(d2.shape)
    idx_flat = d2.argmin(-1)
    # print(idx_flat)
    print(pos.shape)
    print(idx_flat.shape)
    # plt.show()
    np.save('data/pretrain/pos_val.npy', pos)
    np.save('data/pretrain/targets_val.npy', idx_flat)