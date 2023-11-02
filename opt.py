import cv2
import json
import numpy as np
import time
import torch
from pathlib import Path
from scipy.spatial.distance import cdist
from tqdm import tqdm
from ultralytics import YOLO
import ddnet

# Constants
DEVICE = torch.device("mps")
MODEL_PATH = "./save_model/jhmdb.pt"
VIDEO_PATH = "1.mov"
OUTPUT_PATH = "output.mp4"
THRESHOLD = 0.6
LABELS = [
    "brush_hair", "catch", "clap", "climb_stairs", "golf", "jump", 
    "kick_ball", "pick", "pour", "pull-up", "push", "run", "shoot_ball", 
    "shoot_bow", "shoot_gun", "sit", "stand", "swing_baseball", "throw", 
    "walk", "wave"
]
offset = torch.tensor([0, 20])

def norm_scale(x):
    return (x - np.mean(x)) / np.mean(x)

def get_CG(p, C):
    M = [cdist(p[f], p[f], 'euclidean')[np.triu_indices(C.joint_n, 1, C.joint_n)] for f in range(C.frame_l)]
    return norm_scale(np.stack(M))

def compute_new_keypoints(pose_estimations):
    keypoints = np.zeros((15, 2))
    keypoints[0] = (pose_estimations[5] + pose_estimations[6]) / 2
    keypoints[1] = (pose_estimations[11] + pose_estimations[12]) / 2 - offset
    keypoints[2] = torch.mean(pose_estimations[:5], dim=0)
    keypoints[3:] = pose_estimations[[6, 5, 12, 11, 8, 7, 14, 13, 10, 9, 16, 15]]
    return keypoints

def data_generator_rt(T):
    return np.stack([T[i] for i in range(len(T))])

# Initialize models
model = YOLO("yolov8m-pose.pt", task="pose")
j_config = ddnet.JConfig()
ddnet_model = ddnet.DDNet(j_config.frame_l, j_config.joint_n, j_config.joint_d, j_config.feat_d, j_config.filters, j_config.clc_num)

with torch.no_grad():
    ddnet_model.load_state_dict(torch.load(MODEL_PATH))
    ddnet_model.to(DEVICE)
    ddnet_model.eval()

# Video processing
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

sequence_pose = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame, pose=True, conf=0.5)
    if result[0] and result[0].keypoints:
        kp = compute_new_keypoints(result[0].keypoints.xy[0])
        sequence_pose.append(kp)
        for i, (x, y) in enumerate(kp):
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(frame, f"({i})", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if len(sequence_pose) == j_config.frame_l:
        gc = get_CG(sequence_pose, j_config)
        print(gc)
        break
        res = ddnet_model(torch.from_numpy(data_generator_rt(gc)).float().to(DEVICE), torch.from_numpy(data_generator_rt(np.array(sequence_pose))).float().to(DEVICE))
        sequence_pose.pop(0)

        # Display top 5 predictions
        res_numpy = res.cpu().detach().numpy().flatten()
        for i, index in enumerate(np.argsort(res_numpy)[::-1][:5]):
            cv2.putText(frame, f"{i+1}. {LABELS[index]}", (50, 50 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Frame', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
out.release()


# frontend - rtsp stream - backend 
# deep learning: ddnet datasets training and running on jetson nano 
#  1. upload on agx 
#  2. action recognition 
#  3. live stream on vlc 
#  live demo in the lab
#  4. kt


# yerkebulan: frontend : ui improve & keep improving backend
#  maskcam study 

#  aibek: rtsp stream multistream

