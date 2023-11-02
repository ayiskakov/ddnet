import numpy as np
from scipy.spatial.distance import cdist
import copy
import scipy.ndimage.interpolation as inter
from scipy.signal import medfilt


def norm_scale(x):
    return (x-np.mean(x)) / np.std(x)
  
def get_CG(p,C):
    M = []
    iu = np.triu_indices(C.joint_n,1,C.joint_n)
    for f in range(C.frame_l): 
        d_m = cdist(p[f],p[f],'euclidean')       
        d_m = d_m[iu] 
        M.append(d_m)
    M = np.stack(M) 
    M = norm_scale(M)
    return M

def zoom(p, target_l=64, joints_num=25, joints_dim=3):
    p_copy = copy.deepcopy(p)
    l = p_copy.shape[0]
    p_new = np.empty([target_l, joints_num, joints_dim])
    for m in range(joints_num):
        for n in range(joints_dim):
            p_copy[:, m, n] = medfilt(p_copy[:, m, n], 3)
            p_new[:, m, n] = inter.zoom(p_copy[:, m, n], target_l/l)[:target_l]
    return p_new


def compute_new_keypoints(pose_estimations):
    keypoints = np.zeros(shape=(15, 2))
    # Compute the neck as the midpoint between left and right shoulder
    keypoints[0] = (
        (pose_estimations[5] + pose_estimations[6]) / 2
    )
    

    # Compute the belly as the midpoint between left and right hip
    keypoints[1] = (
        (pose_estimations[11] + pose_estimations[12]) / 2 
    )
    keypoints[1][1] -= 20

    # Compute the head using the average of nose, eyes, and ears
    keypoints[2] = (
        pose_estimations[0] +
        pose_estimations[1] +
        pose_estimations[2] +
        pose_estimations[3] +
        pose_estimations[4]
    ) / 5

    # Map the rest directly from pose_estimations
    keypoints[3] = pose_estimations[6] # right_shoulder
    keypoints[4] = pose_estimations[5] # left_shoulder
    keypoints[5] = pose_estimations[12] # right_hip
    keypoints[6] = pose_estimations[11] # left_hip
    keypoints[7] = pose_estimations[8] # right_elbow
    keypoints[8] = pose_estimations[7] # left_elbow
    keypoints[9] = pose_estimations[14] # right_knee
    keypoints[10] = pose_estimations[13] # left_knee
    keypoints[11] = pose_estimations[10] # right_wrist
    keypoints[12] = pose_estimations[9] # left_wrist
    keypoints[13] = pose_estimations[16] # right_ankle
    keypoints[14] = pose_estimations[15]  # left_ankle

    return keypoints


def data_generator(T, C, le):
    X_0 = []
    X_1 = []
    Y = []

    T = np.expand_dims(T, axis = 0)
    for i in range(len(T)): 
        p = T["pose"][i]
        p = zoom(p,target_l=C.frame_l,joints_num=C.joint_n,joints_dim=C.joint_d)
    
        M = get_CG(p,C)
        label = np.zeros(C.clc_num)
        label[T['label']] = 1  
#         X_0.append(M)
#         p = norm_train2d(p)

        X_0.append(M)
        X_1.append(p)
        Y.append(label)

    X_0 = np.stack(X_0)  
    X_1 = np.stack(X_1) 
    Y = np.stack(Y)

    return X_0,X_1,Y