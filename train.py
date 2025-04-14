from ultralytics import YOLO
import torch

DATA_YAML_PATH = 'config/data.yaml'
EPOCHS = 50
IMG_SIZE = 320
BATCH = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device: ', DEVICE)
NAME = 'elderly_fall_detection_cpu'

model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, etc.

results = model.train(
    data=DATA_YAML_PATH,
    epochs=EPOCHS,  
    imgsz=IMG_SIZE,  
    batch=BATCH,    
    device=DEVICE,  
    name=NAME
)

# Evaluate the model
metrics = model.val()
print(metrics)