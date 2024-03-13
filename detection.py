from ultralytics import YOLO

model = YOLO("yolov9e.pt")

classList = [i for i in range(0,10)] # first ten classes

results = model.track(source="enhancer_scenario1_Video.mp4", show=True, device = 'cuda:0')