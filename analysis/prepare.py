# import numpy as np
# from scipy.spatial.distance import cdist
# import copy
# import scipy.ndimage.interpolation as inter
# from scipy.signal import medfilt
from ultralytics import YOLO
import config
import utils
from sklearn import preprocessing

# import prepocessing


# model = YOLO("./yolo/yolov8m-pose.pt", task="pose")


# import cv2
# import os

# conf = config.Config()


# if __name__ == "__main__":
#     datasets_directory = "./datasets"
#     action_directory = os.listdir(datasets_directory)


#     for action in action_directory:
#         if action == ".DS_Store":
#             continue
#         directory = os.path.join(datasets_directory, action)
#         action_videos = os.listdir(directory)

#         processed_data_directory = "./processed_data"
#         if not os.path.exists(processed_data_directory):
#             os.makedirs(processed_data_directory)

#         for video in action_videos:
#             # print(video)
#             video_path = os.path.join(directory, video)
#             # print(video_path)
#             cap = cv2.VideoCapture(video_path)

#             # we will accumulate pose estimations on each frame according to conf.frame_l
#             # while performing pose estimation on each frame
#             # and then we will calculate JCD features for each frame after accumulating conf.frame_l frames
#             # then we will store results in a files for future training and testing

            
#             frames_count = 0
#             sequence_pose = []
            

#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                
#                 result = model(frame, pose=True)

#                 if result[0] is not None:
#                     if result[0].keypoints is not None:
#                         #  get keypoints
#                         keypoints = result[0].keypoints
#                         # annotate frame with keypoints
#                         # frame = keypoints.draw(frame, radiбus=5, thickness=2)
#                         #  draw keypoints
#                         kp = utils.compute_new_keypoints(keypoints.xy[0])

#                         sequence_pose.append(kp)
                



#                 if len(sequence_pose) == conf.frame_l:
#                     #  compute GC
#                     gc = utils.get_CG(sequence_pose, conf)

#                     save_data = {
#                         'action': action,
#                         'video_name': video,
#                         'sequence_pose': sequence_pose,
#                         'gc': gc
#                     }
#                     file_name = f"{action}_{video.split('.')[0]}_data.npy"
#                     file_path = os.path.join(processed_data_directory, file_name)
#                     np.save(file_path, save_data)

#                     # we will store sequence pose and gc in a file as separate variables
#                     # then we will use this file for training and testing

#                     break




            
#             cap.release()
#             cv2.destroyAllWindows()


import numpy as np
import os
import cv2
from concurrent.futures import ProcessPoolExecutor


label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(["boxing", "handwaving", "walking"])


conf = config.Config()
model = YOLO("./yolo/yolov8m-pose.pt", task="pose")

def process_video(args):
    action, video, directory, conf = args
    video_path = os.path.join(directory, video)
    cap = cv2.VideoCapture(video_path)

    sequence_pose = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, pose=True)

        if result[0] is not None and result[0].keypoints is not None and len(result[0].keypoints.xy[0]) == 17:
            keypoints = result[0].keypoints
            kp = utils.compute_new_keypoints(keypoints.xy[0])
            sequence_pose.append(kp)

        if len(sequence_pose) == conf.frame_l:
            gc = utils.get_CG(sequence_pose, conf)

            # Return the data instead of saving
            return {
                'action': action,
                'video_name': video,
                'sequence_pose': sequence_pose,
                'gc': gc
            }

    cap.release()
    cv2.destroyAllWindows()
    return None

if __name__ == "__main__":
    datasets_directory = "./datasets"
    action_directory = os.listdir(datasets_directory)

    processed_data_directory = "./processed_data"
    if not os.path.exists(processed_data_directory):
        os.makedirs(processed_data_directory)

    tasks = []

    for action in action_directory:
        if action == ".DS_Store":
            continue
        directory = os.path.join(datasets_directory, action)
        action_videos = os.listdir(directory)

        for video in action_videos:
            tasks.append((action, video, directory, conf))

    # Use parallel processing to speed up the video processing
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_video, tasks))

    for data in results:
        if data:
            file_name = f"{data['action']}_{data['video_name'].split('.')[0]}_data.npy"
            file_path = os.path.join(processed_data_directory, file_name)
            np.save(file_path, data)
