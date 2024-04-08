import numpy as np

intrinsic = np.array([[3451.5, 0.0, 2312],
                      [0.0, 3451.5, 1734],
                      [0.0, 0.0, 1.0]])

np.save('Data\intrinsic.npy', intrinsic)