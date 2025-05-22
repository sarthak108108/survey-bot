import sys
import os

yolo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'yolov3'))
sys.path.insert(0, yolo_path)

import torch
from models.experimental import attempt_load
from utils.dataloaders import letterbox
from utils.general import non_max_suppression, scale_boxes
import cv2
import numpy as np

weights = 'yolov3.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights)
model.eval()

def Get_Predictions(image_path):
    img0 = cv2.imread(image_path)  # Original image
    img = letterbox(img0, new_shape=640)[0]  # Resize
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    if img.ndimension() == 3:
         img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, 0.25, 0.45)

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                label_clean = model.names[int(cls)]
                print(f'Detected {label} at {xyxy}')
            # Draw on image
                cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    output_file_name = image_path[:-5] + '_out.jpg'

    cv2.imwrite(output_file_name, img0)
    return output_file_name, label_clean
