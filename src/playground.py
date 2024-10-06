import numpy as np

a = np.array([[0],[1]])
b = np.copy(a[0])
a[0,0]=1
print(b)