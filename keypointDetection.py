import io
from PIL import Image
import numpy as np
import PIL
import requests
import torch
import openpifpaf
import cv2
import matplotlib.pyplot as plt
import os

from keypointPlotter import KeyPointPlotter
from openpifpaf.show.painters import KeypointPainter, CrowdPainter, DetectionPainter

video_name = "Albany@GeorgeNorth.mp4"
output_video_name = "Albany@GeorgeNorth_CarKp.mp4"


cap = cv2.VideoCapture(video_name)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_name, fourcc=fourcc, fps=10, frameSize=(frame_width,frame_height))

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

success = True
num_frame = 0

PAINTERS = {
    'Annotation': KeypointPainter(line_width=1.0),
    'AnnotationCrowd': CrowdPainter,
    'AnnotationDet': DetectionPainter,
}

# predictor = openpifpaf.Predictor(checkpoint= 'shufflenetv2k16')
predictor = openpifpaf.Predictor(checkpoint= 'shufflenetv2k30-apollo-66')

annotation_painter = openpifpaf.show.AnnotationPainter(painters=PAINTERS)
while(cap.isOpened()):
    success, frame = cap.read()

    if success:
        color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(color_coverted)
        im = np.asarray(img)
        with openpifpaf.show.image_canvas(im) as ax:
            pass

        
        predictions, gt_anns, image_meta = predictor.pil_image(img)
        # print(predictions)

        
        with KeyPointPlotter.image(im) as ax:
            annotation_painter.annotations(ax, predictions)

         # Convert image with keypoints back to OpenCV format
        buf = io.BytesIO()
        ax.figure.savefig(buf, format='png')
        buf.seek(0)
        img_with_keypoints = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), 1)

        # Resize image to match frame size
        img_with_keypoints_resized = cv2.resize(img_with_keypoints, (frame_width, frame_height))

        # Write frame to output video
        out.write(img_with_keypoints_resized)
        # cv2.imshow('IMG',img_with_keypoints_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print(num_frame)
        num_frame += 1

        if num_frame > 500:
            break
        

    else:
        break


cap.release()
out.release()
cv2.destroyAllWindows()



    
    


    
