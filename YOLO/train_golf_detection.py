from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO('models/yolov8s.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data='datasets/Segmentation-Batch-16-1/data.yaml', epochs=10, imgsz=1280)

if __name__ == '__main__':
    main()
