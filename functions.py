import cv2
import numpy as np

sift = cv2.SIFT_create()

def extract_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def match_features(base_descriptors, new_descriptors):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(base_descriptors, new_descriptors, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    return good_matches

def reproject_to_image_plane(point_3d, R, t, K):
    t = t.reshape(-1)
    point_image_plane = K @ (R @ point_3d + t)
    point_image_plane /= point_image_plane[2]
    return point_image_plane[:2]

def estimate_camera_pose(K, new_keypoints, new_descriptors, base_keypoints, base_descriptors, base_3D_points, matches):
    # base image의 matches
    query_idx = [match.queryIdx for match in matches]
    # new image의 matches
    train_idx = [match.trainIdx for match in matches]

    base_matched_keypoints = np.zeros([len(query_idx), 2])
    new_matched_keypoints = np.zeros([len(train_idx), 2])
    base_matched_descriptors = np.zeros([len(query_idx), 128])
    new_matched_descriptors = np.zeros([len(train_idx), 128])

    matched_3D_points = np.zeros([len(train_idx), 3])

    for i, idx in enumerate(query_idx):
        base_matched_keypoints[i] = base_keypoints[idx]
        base_matched_descriptors[i] = base_descriptors[idx]
        matched_3D_points[i] = base_3D_points[idx]

    for i, idx in enumerate(train_idx):
        new_matched_keypoints[i] = new_keypoints[idx].pt
        new_matched_descriptors[i] = new_descriptors[idx]

    best_pose = None
    best_inliers_count = 0

    for i in range(1000):
        indices = np.random.choice(new_matched_keypoints.shape[0], 3, replace=False)
        points_3D = matched_3D_points[indices]
        points_2D = new_matched_keypoints[indices]

        retval, rvecs, tvecs = cv2.solveP3P(points_3D, points_2D, K, None, flags=cv2.SOLVEPNP_AP3P)
        if retval:
            for rvec, tvec in zip(rvecs, tvecs):
                rvec, _ = cv2.Rodrigues(rvec)

                inliers_count = 0
                inliers = []

                for j in range(new_matched_keypoints.shape[0]):
                    point_2d = new_matched_keypoints[j] 
                    point_3d = matched_3D_points[j]
                    reprojected_point = reproject_to_image_plane(point_3d, rvec, tvec, K)
                    error = np.sqrt(np.sum(np.square(point_2d - reprojected_point)))
                    if error < 100:
                        inliers_count += 1
                        inliers.append(j)
                
                if inliers_count > best_inliers_count:
                    best_inliers_count = inliers_count
                    best_pose = np.hstack((rvec, tvec))

    return best_pose

def reconstruct_3D_points(K, base_camera_pose, new_camera_pose, base_keypoints, new_keypoints, matches):

    query_idx = [match.queryIdx for match in matches]
    train_idx = [match.trainIdx for match in matches]
    
    invk = np.linalg.inv(K)
    base_2D_points = np.float32([base_keypoints[m.queryIdx].pt for m in matches])
    new_2D_points = np.float32([new_keypoints[m.trainIdx].pt for m in matches])

    ones = np.ones((base_2D_points.shape[0], 1))
    base_points = np.hstack((base_2D_points, ones))
    new_points = np.hstack((new_2D_points, ones))

    normalized_base_points = (invk @ base_points.T).T
    normalized_new_points = (invk @ new_points.T).T

    P1 = base_camera_pose
    P2 = new_camera_pose

    points_3D = np.zeros((base_2D_points.shape[0], 3))

    for i, (base_point, new_point) in enumerate(zip(normalized_base_points, normalized_new_points)):
        A = np.zeros((4, 4))
        A[0] = base_point[0] * P1[2] - P1[0]
        A[1] = base_point[1] * P1[2] - P1[1]
        A[2] = new_point[0] * P2[2] - P2[0]
        A[3] = new_point[1] * P2[2] - P2[1]

        _, _, Vt = np.linalg.svd(A)
        points_3D[i] = Vt[-1, :3] / Vt[-1, 3]
    
    return query_idx, train_idx, base_2D_points, new_2D_points, points_3D

def is_valid_3D(K, base_camera_pose, query_idx, base_matched_idx, base_3D_points, base_2D_points, points_3D):
    valid_3D_points = []
    inlinear = []
    R = base_camera_pose[:3, :3] 
    t = base_camera_pose[:3, 3]  

    for i, idx in enumerate(query_idx):
        if idx in base_matched_idx:
            j = np.where(base_matched_idx == idx)
            # idx가 base_matched_idx의 몇번째에 위치하는지 확인하고 j번째에 위치하면 base_3D_points[k]를 구함
            error = np.sqrt(np.sum(np.square(points_3D[i] - base_3D_points[j[0][0]].T)))
            if error < 25:
                valid_3D_points.append(base_3D_points[j[0][0]].T)
                inlinear.append(i)

        else:
            reprojected_to_2D = (R @ points_3D[i].T + t)
            reprojected_to_2D = K @ reprojected_to_2D
            reprojected_to_2D /= reprojected_to_2D[2]
            error = np.sqrt(np.sum(np.square(reprojected_to_2D[:2] - base_2D_points[i])))
            if error < 100:
                valid_3D_points.append(points_3D[i])
                inlinear.append(i)

    new_inlinear = np.array(inlinear)
    new_3D_points = np.float32(valid_3D_points)

    return new_inlinear, new_3D_points

    
                
