from matplotlib.pyplot import annotate
from ultralytics import YOLO
from pathlib import Path
import cv2

# Load a model
model = YOLO('models/golf_segmentation.pt')

img_path = Path(r'D:\Projects\RealSense\data\30_bags\20240923_145854_bag')

# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] *
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

for img_file in img_path.iterdir():
    # Perform object detection on an image
    results = model(str(img_file))

    # Retrieve the original image
    img = results[0].orig_img

    for result in results:
        # get the classes names
        classes_names = result.names

        # iterate over each box
        for box in result.boxes:
            # check if confidence is greater than 40 percent
            if box.conf[0] > 0.4:
                # get coordinates
                [x1, y1, x2, y2] = box.xyxy[0]
                # convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # get the class
                cls = int(box.cls[0])

                # get the class name
                class_name = classes_names[cls]

                # get the respective colour
                colour = getColours(cls)

                # draw the rectangle on the original image
                cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)

                # put the class name and confidence on the image
                cv2.putText(img, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

    # show the image
    cv2.imshow('img', img)

    # break the loop if 'q' is pressed
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()