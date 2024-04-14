import io
from PIL import Image
import numpy as np
import PIL
import requests
import torch
import openpifpaf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from keypointPlotter import KeyPointPlotter

video_name = "Highway_clip.mp4"
output_video_name = "Highway_clip_Keypoints1.mp4"


cap = cv2.VideoCapture(video_name)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter(output_video_name, fourcc=fourcc, fps=8, frameSize=(frame_width,frame_height))

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

success = True
num_frame = 0

predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16-apollo-24')

while(cap.isOpened()):
    success, frame = cap.read()

    if success:
        color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(color_coverted)
        im = np.asarray(img)
        with openpifpaf.show.image_canvas(im) as ax:
            pass

        
        predictions, gt_anns, image_meta = predictor.pil_image(img)
        print(predictions.data[0])

        # annotation_painter = openpifpaf.show.AnnotationPainter()
        # with KeyPointPlotter.image(im) as ax:
        #     annotation_painter.annotations(ax, predictions)

        # buf = io.BytesIO()
        # KeyPointPlotter.fig.savefig(buf, format='png')
        # buf.seek(0)

        # # Create a PIL image from the byte string
        # img = Image.open(buf)
        # # Display the image
        # # img.show()

        # # Convert PIL image to NumPy array
        # np_image = np.array(img)
        
        # # Convert from RGB to BGR (OpenCV uses BGR)
        # bgr_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        
        
        # # Write the frame to the video file
        # out.write(bgr_image)
        # #cv2.imshow('IMG',bgr_image)
        print(num_frame)
        num_frame += 1

        if num_frame > 200:
            break
        

    else:
        break


cap.release()
#out.release()



    
    


    
