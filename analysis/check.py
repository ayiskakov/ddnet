import cv2
import os


# we have following directories with videos ./datasets/boxing, ./datasets/hand-waving, ./datasets/walking
# first we need to extract the frames from the videos and check if all videos has 60 frames
# we can use ffmpeg to extract the frames from the videos

if __name__ == "__main__":
    # open directory with videos
    # for each video
    # extract the frames
    # check if all videos has 60 frames
    # if not, print not 60 frames
    # if yes, print 60 frames

    datasets_directory = "./datasets"
    action_directory = os.listdir(datasets_directory)


    for action in action_directory:
        if action == ".DS_Store":
            continue
        directory = os.path.join(datasets_directory, action)
        action_videos = os.listdir(directory)
        
        count = 0
        not_60_frames = 0

        for video in action_videos:
            # print(video)
            video_path = os.path.join(directory, video)
            # print(video_path)
            cap = cv2.VideoCapture(video_path)
            # print(cap)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # print(frame_count)
            count += 1
            if frame_count < 60:
                not_60_frames += 1

            cap.release()
            cv2.destroyAllWindows()

        print("Action:", action)
        print("\tTotal videos: ", count, " Videos with less than 60 frames: ", not_60_frames)
        print()


