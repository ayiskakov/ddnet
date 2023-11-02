from ultralytics import YOLO
import json
import cv2
import time
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
import ddnet
from pathlib import Path
import torch

model = YOLO("yolov8m-pose.pt", task="pose")

def norm_scale(x):
    return (x-np.mean(x))/np.std(x)
  
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


# results = model("1.mov", show=True, pose=True, conf=0.5)

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

class Config():
    def __init__(self):
        self.frame_l = 60  # the length of frames
        self.joint_n = 15  # the number of joints
        self.joint_d = 2  # the dimension of joints
        self.clc_num = 3  # the number of class
        self.feat_d = 105
        self.filters = 64

j_config = Config()
ddnet_model = ddnet.DDNet(j_config.frame_l, j_config.joint_n, j_config.joint_d, j_config.feat_d, j_config.filters, j_config.clc_num)
save_model_dir = Path("./save_model")

device = torch.device("cpu")

ddnet_model.load_state_dict(torch.load(Path(save_model_dir/Path("ours.pt"), map_location=device)))

ddnet_model.to(device)
ddnet_model.eval()


#  open video file with opencv
cap = cv2.VideoCapture("1.mov")

#  get video file properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

ret, frame = cap.read()

time0 = 0
sequence_pose = []
sequence_gc = []
predictions = []
threshold = 0.6

def data_generator_rt(T):
    X = []

    T = np.expand_dims(T, axis = 0)
    for i in range(len(T)): 
        p = T[i]
#         p = zoom(p,target_l=C.frame_l,joints_num=C.joint_n,joints_dim=C.joint_d)

#         M = get_CG(p,C)

#         X_0.append(M)
#         p = norm_train2d(p)

        X.append(p)

    X = np.stack(X)  
#     X_1 = np.stack(X_1) 

    return X

# record a video locally with opencv in mp4 format
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

#  record a video locally with opencv in mp4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))


#  brush hair, catch, clap, climb stairs, golf, jump, kick ball, pick, pour, pull-up, push, run, shoot ball, shoot bow, shoot gun, sit, stand, swing baseball, throw, walk, wave
labels_text = ["boxing", "hand-waving", "walking"]
while(cap.isOpened()):
    #  read one frame
    ret, frame = cap.read()
    if ret == False:
        break

    #  do object detection
    result = model(frame, pose=True, conf=0.5)

    if result[0] is not None:
        if result[0].keypoints is not None:
            #  get keypoints
            keypoints = result[0].keypoints
            # annotate frame with keypoints
            # frame = keypoints.draw(frame, radiбus=5, thickness=2)
            #  draw keypoints
            kp = compute_new_keypoints(keypoints.xy[0])
            #  draw skeleton
            sequence_pose.append(kp)


            for i, (x, y) in enumerate(kp):

                # Draw a circle at each keypoint
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

                # Annotate the keypoint with its coordinates
                cv2.putText(frame, f"({i})", (int(x), int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)



    if len(sequence_pose) == j_config.frame_l:
        #  compute GC
        gc = get_CG(sequence_pose, j_config)
        sequence_gc.append(gc)

        res = ddnet_model(torch.from_numpy(data_generator_rt(gc)).float().to(device), torch.from_numpy(data_generator_rt(np.array(sequence_pose))).float().to(device))
        sequence_pose = sequence_pose[1:]
        print(res, "ddnet")
        # print(labels_text[np.argmax(res.cpu().detach().numpy())])
        # annotate frame with labels_text
        # cv2.putText(frame, labels_text[np.argmax(res.cpu().detach().numpy())], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # display top 5 argmax predictions
        # Assuming res and labels_text are defined
        res_numpy = res.cpu().detach().numpy().flatten()  # Ensure res is a 1D numpy array
        sorted_indices = np.argsort(res_numpy)[::-1]
        top_5_indices = sorted_indices[:5]  # Get the indices of the top 5 values

        # Now iterate through these indices and display the labels
        y_position = 50  # Initial y position
        for i, index in enumerate(top_5_indices):
            print(index)
            label_text = labels_text[index]
            cv2.putText(frame,str(i+1)+ ". " +label_text, (50, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            y_position += 40  # Increment the y position for the next label
        


    #  show frame
    cv2.imshow('Frame', frame)
    out.write(frame)

    #  press `q` to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  release opencv objects
cap.release()
cv2.destroyAllWindows()

#  close output video file
out.release()



# 0: nose
# 1: left_eye
# 2: right_eye
# 3: left_ear
# 4: right_ear
# 5: left_shoulder
# 6: right_shoulder
# 7: left_elbow
# 8: right_elbow
# 9: left_wrist
# 10: right_wrist
# 11: left_hip
# 12: right_hip
# 13: left_knee
# 14: right_knee
# 15: left_ankle
# 16: right_ankle
