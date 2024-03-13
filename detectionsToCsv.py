from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov9c.pt')
classList = [i for i in range(0,10)]

# Open the video file
video_path = "enhancer_scenario1_Video.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Dataframe to convert to csv
column_names =['frame','id','bb_left', 'bb_top', 'bb_width', 'bb_height']
df = pd.DataFrame(columns=column_names)

num_frames = 0 # track frames
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        num_frames += 1
        results = model.track(frame, persist=True,device = 'cuda:0', classes=classList)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box

            # find top left point of bounding box
            x = float(x)-float(w)/2
            y = float(y)-float(h)/2
            # append to dataframe
            df.loc[len(df.index)] = [num_frames,track_id,float(x),float(y),float(w),float(h)]
            #track = track_history[track_id]
            #track.append((float(x), float(y)))  # x, y center point
            #if len(track) > 30:  # retain 90 tracks for 90 frames
                #track.pop(0)

            # Draw the tracking lines
            #points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            #cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Display the annotated frame
        #cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        print(num_frames)
        if num_frames > 3000:
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

# save to csv
df = df.astype(int)
df.to_csv('enhancer_scenario1_Video_YOLO9.csv', index=False)