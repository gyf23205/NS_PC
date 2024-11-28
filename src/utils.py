import numpy as np

def dm_to_array(dm):
        return np.array(dm.full())

def align_length(x: list, length):
    for i in range(len(x)):
          x[i] = np.pad(x[i], ((0, 0), (0, 0), (0, length-x[i].shape[2])), mode='edge')
    pass

def action2waypoints(actions, world_size, len_grid):
    n_col = world_size[1]
    row = actions // n_col
    col = actions % n_col
    coord_x = 0.5 * len_grid + col * len_grid
    coord_y = 0.5 * len_grid + row * len_grid
    return np.array([coord_x, coord_y])
    # return np.concat([coord_x, coord_y], axis=-1)


if __name__=='__main__':
      length = 15
      x = [np.random.random((3, 21, 10)), np.random.random((3, 21, 15))]
      align_length(x, 15)
      for a in x:
            print(a.shape)