from ultralytics import YOLO
import cv2

# model = YOLO('models/yolov8x.pt')
model = YOLO('models/golf_segmentation.pt')
result = model.predict('inputs/03_face_on_view.mp4', save=True)
# print(result)
# for box in result[0].boxes:
#     print(box)
