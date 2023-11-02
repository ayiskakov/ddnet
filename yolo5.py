import cv2
import torch

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Webcam index (usually 0 or 1 depending on your setup)
webcam_index = 0

# Create a VideoCapture object
cap = cv2.VideoCapture(webcam_index)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # Make detections
    results = model(frame)

    # Render the frame with bounding boxes
    results.render()  # updates results.imgs with boxes and labels
    for img in results.ims:
        # Display the resulting frame
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('YOLOv5 Object Detection', img)
        if cv2.waitKey(1) == ord('q'):  # quit on 'q' key press
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
