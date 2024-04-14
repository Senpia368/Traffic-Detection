import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
import pandas as pd

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# im = cv2.imread('screenshot1.png')

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.DEVICE = 'cuda'
predictor = DefaultPredictor(cfg)
#outputs = predictor(im)

cap = cv2.VideoCapture('enhancer_scenario1_Video.mp4')

# Dataframe to convert to csv
column_names =['frame','id','bb_left', 'bb_top', 'bb_width', 'bb_height']
df = pd.DataFrame(columns=column_names)

num_frames = 0 # track frames

while cap.isOpened():
    success, frame = cap.read()
    if success:
        num_frames += 1
        outputs = predictor(frame)
        for box in outputs['instances'].pred_boxes:
            x,y,w,h = box
            df.loc[len(df.index)] = [num_frames,num_frames,float(x),float(y),float(w),float(h)]

        #v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #cv2.imshow('output',out.get_image()[:, :, ::-1])
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if num_frames > 1000:
            break
        print(num_frames)
    else:
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

# save to csv
df = df.astype(int)
df.to_csv('enhancer_scenario1_Video_Detectron2.csv', index=False)