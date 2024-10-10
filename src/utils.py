import numpy as np

def dm_to_array(dm):
        return np.array(dm.full())

def align_length(x: list, length):
    for i in range(len(x)):
          x[i] = np.pad(x[i], ((0, 0), (0, 0), (0, length-x[i].shape[2])), mode='edge')
    pass

if __name__=='__main__':
      length = 15
      x = [np.random.random((3, 21, 10)), np.random.random((3, 21, 15))]
      align_length(x, 15)
      for a in x:
            print(a.shape)