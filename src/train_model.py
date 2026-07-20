from ultralytics import YOLO

def main():
    # Load a pretrained YOLOv8 Nano Model
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')

    # Train the model
    print("Start training process...")
    results = model.train(
        data='data.yaml', 
        epochs=100, 
        imgsz=640, 
        batch=16, 
        device='cpu',
        workers=4, 
        project='runs/train', 
        name='yolov8n_custom',
        exist_ok=True)

    print("Training completed. Results saved in 'runs/train/yolov8n_custom' directory.")

if __name__ == "__main__":
    main()