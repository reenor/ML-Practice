from ultralytics import YOLO
import cv2

# model = YOLO('models/yolov8x.pt')
model = YOLO('models/golf_best_2.pt')
result = model.predict('inputs/golf_video.mp4', save=True)
print(result)
for box in result[0].boxes:
    print(box)
