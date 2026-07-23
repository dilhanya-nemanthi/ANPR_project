import os
import cv2
from ultralytics import YOLO

def main():
    print("--- Script Started ---")
    
    # 1. Check if the model exists
    model_path = r"runs\detect\GPU_Results\training_run_1-3"
    if not os.path.exists(model_path):
        print(f"ERROR: Cannot find model at {model_path}")
        return

    # 2. Check if the test image exists
    image_path = "test_car.avif" 
    if not os.path.exists(image_path):
        print(f"ERROR: Cannot find image at {image_path}. Did you name it correctly and place it in the ANPR_project folder?")
        return

    print("Model and image found! Loading YOLO...")
    model = YOLO(model_path)

    print(f"Running detection on {image_path}...")
    results = model.predict(source=image_path, save=True)

    print("Opening window...")
    annotated_img = results[0].plot()
    cv2.imshow("License Plate Detection", annotated_img)
    print("Window opened! Click on the image window and press ANY KEY to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()