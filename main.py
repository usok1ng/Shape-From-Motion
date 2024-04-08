# Import some necessary libs
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from functions import extract_features, match_features, estimate_camera_pose, reconstruct_3D_points, is_valid_3D

K = np.load("Data/intrinsic.npy")

sfm03_camera_pose = np.load("two_view_recon_info/sfm03_camera_pose.npy")
sfm04_camera_pose = np.load("two_view_recon_info/sfm04_camera_pose.npy")

initial_3D_points = np.load("two_view_recon_info/3D_points.npy")

base_sfm03_keypoints = np.load("two_view_recon_info/base_sfm03_keypoints.npy")
base_sfm03_descriptors = np.load("two_view_recon_info/base_sfm03_descriptors.npy")
base_sfm03_matched_idx = np.load("two_view_recon_info/base_sfm03_matched_idx.npy")
base_sfm04_keypoints = np.load("two_view_recon_info/base_sfm04_keypoints.npy")
base_sfm04_descriptors = np.load("two_view_recon_info/base_sfm04_descriptors.npy")
base_sfm04_matched_idx = np.load("two_view_recon_info/base_sfm04_matched_idx.npy")

# Image paths sorted by filename
image_paths = sorted(glob.glob("Data/sfm*.jpg"), key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace('sfm', '')))

# Ordered sequence of image indices to process
image_sequence = [2, 1, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

# Initialize SIFT feature detector
sift = cv2.SIFT_create()

# 이전 이미지의 특징점과 디스크립터를 초기화합니다.
prev_keypoints, prev_descriptors, prev_matched_idx = None, None, None

# 카메라 포즈 초기화
prev_camera_pose = np.eye(4)

# 3D 포인트 배열을 초기화합니다.
all_3D_points = initial_3D_points
base_3D_points = None

# Main Loop for step 1
for idx in image_sequence:
    # Get the path of the new image to process
    new_image_path = image_paths[idx]
    print(f"Processing image: {new_image_path}")

    if idx == 2:
        base_keypoints, base_descriptors, base_matched_idx = base_sfm03_keypoints, base_sfm03_descriptors, base_sfm03_matched_idx
        base_camera_pose = sfm03_camera_pose
        base_image_path = image_paths[3]
        base_3D_points = initial_3D_points
        
    elif idx == 5:
        base_keypoints, base_descriptors, base_matched_idx = base_sfm04_keypoints, base_sfm04_descriptors, base_sfm04_matched_idx
        base_camera_pose = sfm04_camera_pose
        base_image_path = image_paths[4]
        base_3D_points = initial_3D_points

    else:
        base_keypoints, base_descriptors, base_matched_idx = prev_keypoints, prev_descriptors, prev_matched_idx
        base_camera_pose = prev_camera_pose
        base_image_path = image_paths[idx+1] if idx < 3 else image_paths[idx-1]
        base_3D_points = prev_3D_points
    
    # Extract features for the new image
    new_keypoints, new_descriptors = extract_features(new_image_path)

    # Match features between the base image and the new image
    good_matches = match_features(base_descriptors, new_descriptors)

    # Estimate camera pose
    new_camera_pose = estimate_camera_pose(K, new_keypoints, new_descriptors, base_keypoints, base_descriptors, base_3D_points, good_matches)

    # 3D Recon
    new_keypoints, new_descriptors = extract_features(new_image_path)
    base_keypoints, base_descriptors = extract_features(base_image_path)
    good_matches = match_features(base_descriptors, new_descriptors)
    query_idx, train_idx, base_2D_points, new_2D_points, recon_3D_points = reconstruct_3D_points(K, base_camera_pose, new_camera_pose, base_keypoints, new_keypoints, good_matches)

    new_inlinear, new_3D_points = is_valid_3D(K, base_camera_pose, query_idx, base_matched_idx, base_3D_points, base_2D_points, recon_3D_points)

    new_index_keypoints = []
    new_index_descriptors = []
    new_matched_idx_3D = []

    for value in new_inlinear:
        new_index_keypoints.append(new_keypoints[value].pt)
        new_index_descriptors.append(new_descriptors[value])
        new_matched_idx_3D.append(train_idx[value])

    next_base_keypoints = np.array(new_index_keypoints)
    next_base_descriptors = np.array(new_index_descriptors)
    next_base_matched_idx = np.array(new_matched_idx_3D)

    prev_3D_points = new_3D_points

    all_3D_points = np.vstack((all_3D_points, new_3D_points))

    prev_keypoints = next_base_keypoints
    prev_descriptors = next_base_descriptors
    prev_matched_idx = next_base_matched_idx
    prev_camera_pose = new_camera_pose
    
    np.save("prev_keypoints", prev_keypoints)
    np.save("prev_descriptors", prev_descriptors)
    np.save("prev_matched_idx", prev_matched_idx)
    np.save("prev_camera_pose", prev_camera_pose)

np.save("all_3D_points", all_3D_points)