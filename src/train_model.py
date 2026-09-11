from ultralytics import YOLO

def train_model():
    # 1. Start from scratch with the base model
    model = YOLO('yolov8n.pt')

    # 2. Train on your fresh, clean dataset with heavy augmentations
    results = model.train(
        data='data.yaml', # Ensure this points to your new dataset folder!
        epochs=50, 
        patience=20, 
        imgsz=640,
        batch=16, 
        device=0, 
        workers=2,
        
        # --- THE DATA AUGMENTATION TRICKS ---
        degrees=3.0,       # Randomly rotates the image by up to 3 degrees
        hsv_v=0.6,         # Randomly drops or raises brightness (simulates night/glare)
        hsv_s=0.5,         # Randomly changes color saturation
        hsv_h=0.015,       # Randomly shifts the color hue slightly
        mosaic=1.0,        # Stitches 4 images together to destroy background memorization
        mixup=0.1,         # Blends two images on top of each other (simulates reflections)
        scale=0.5,         # Randomly zooms in and out (simulates distance)
        
        project='runs/detect',
        name='yolov8n_augmented_scratch' 
    )

if __name__ == '__main__':
    train_model()