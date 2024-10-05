import numpy as np

len_grid = 2
size_world = [3, 4]
x_coord = 0.5 * len_grid + np.repeat(np.arange(size_world[1])[None, :] * len_grid, size_world[0], axis=0)
y_coord = 0.5 * len_grid + np.repeat(np.arange(size_world[0])[:, None] * len_grid, size_world[1], axis=1)
print(x_coord.shape)
print(y_coord.shape)
center_locations = np.sqrt(x_coord**2 + y_coord**2)
print(center_locations.shape)