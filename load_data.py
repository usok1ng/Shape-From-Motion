import numpy as np

# Load initial keypoints, descriptors, and camera poses
sfm03_keypoints, sfm03_descriptors = np.load("two_view_recon_info/sfm03_keypoints.npy"), np.load("two_view_recon_info/sfm03_descriptors.npy")
sfm04_keypoints, sfm04_descriptors = np.load("two_view_recon_info/sfm04_keypoints.npy"), np.load("two_view_recon_info/sfm04_descriptors.npy")
sfm03_camera_pose = np.load("two_view_recon_info/sfm03_camera_pose.npy")
sfm04_camera_pose = np.load("two_view_recon_info/sfm04_camera_pose.npy")

# Load matched indices
sfm03_matched_idx = np.load("two_view_recon_info/sfm03_matched_idx.npy")
sfm04_matched_idx = np.load("two_view_recon_info/sfm04_matched_idx.npy")

# Load initial 3D points
base_3D_points = np.load("two_view_recon_info/3D_points.npy")
inlinear = np.load("two_view_recon_info/inlinear.npy")

# Preprocess using indexing
sfm03_sorted_keypoints = []
sfm03_sorted_descriptors = []
for idx in sfm03_matched_idx:
    sfm03_sorted_keypoints.append(sfm03_keypoints[idx])
    sfm03_sorted_descriptors.append(sfm03_descriptors[idx])

sfm04_sorted_keypoints = []
sfm04_sorted_descriptors = []
for idx in sfm04_matched_idx:
    sfm04_sorted_keypoints.append(sfm04_keypoints[idx])
    sfm04_sorted_descriptors.append(sfm04_descriptors[idx])

sfm03_index_keypoints = []
sfm03_index_descriptors = []
sfm03_matched_idx_3D = []
sfm04_index_keypoints = []
sfm04_index_descriptors = []
sfm04_matched_idx_3D = []

for value in inlinear:
    sfm03_index_keypoints.append(sfm03_sorted_keypoints[value])
    sfm03_index_descriptors.append(sfm03_sorted_descriptors[value])
    sfm03_matched_idx_3D.append(sfm03_matched_idx[value])
    sfm04_index_keypoints.append(sfm04_sorted_keypoints[value])
    sfm04_index_descriptors.append(sfm04_sorted_descriptors[value])
    sfm04_matched_idx_3D.append(sfm04_matched_idx[value])
        
base_sfm03_keypoints = np.array(sfm03_index_keypoints)
base_sfm03_descriptors = np.array(sfm03_index_descriptors)
base_sfm03_matched_idx = np.array(sfm03_matched_idx_3D)
base_sfm04_keypoints = np.array(sfm04_index_keypoints)
base_sfm04_descriptors = np.array(sfm04_index_descriptors)
base_sfm04_matched_idx = np.array(sfm03_matched_idx_3D)

np.save("two_view_recon_info/base_sfm03_keypoints", base_sfm03_keypoints)
np.save("two_view_recon_info/base_sfm03_descriptors", base_sfm03_descriptors)
np.save("two_view_recon_info/base_sfm03_matched_idx", base_sfm03_matched_idx)
np.save("two_view_recon_info/base_sfm04_keypoints", base_sfm04_keypoints)
np.save("two_view_recon_info/base_sfm04_descriptors", base_sfm04_descriptors)
np.save("two_view_recon_info/base_sfm04_matched_idx", base_sfm04_matched_idx)