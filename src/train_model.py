from ultralytics import YOLO

def train_model():
    # 1. Load the pre-trained base model
    model = YOLO('yolov8n.pt')

    # 2. Train with anti-overfitting hyperparameters
    results = model.train(
        data='data.yaml',
        epochs=50,             # Upper limit; early stopping will handle termination
        patience=20,            # Stop training if val loss doesn't improve for 20 epochs
        imgsz=640,
        batch=16,               # Reduce to 8 if you run into GPU memory errors
        device=0,               # Set to 'cpu' if no dedicated NVIDIA GPU is available
        workers=2,
        
        # Augmentations to improve CCTV generalization
        mosaic=1.0,             # Combines 4 images to destroy background memorization
        mixup=0.1,              # Blends images together
        hsv_h=0.015,            # Slight color hue shifts
        hsv_s=0.5,              # Saturation shifts for lighting changes
        hsv_v=0.4,              # Brightness shifts (simulates shadow/glare)
        degrees=10.0,           # Random rotations for tilted plates
        
        project='runs/detect',
        name='yolov8n_anpr_run2'
    )

if __name__ == '__main__':
    train_model()