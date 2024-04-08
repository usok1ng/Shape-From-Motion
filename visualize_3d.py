import numpy as np
import trimesh

all_3D_points = np.load("all_3D_points.npy")
trimesh.Trimesh(vertices=all_3D_points).export('./multiview.ply')