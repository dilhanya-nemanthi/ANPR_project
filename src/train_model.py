import os
from pathlib import Path
from ultralytics import YOLO

def main():
   
   
    
    
    data_yaml_path = Path(r"C:\Users\User\Desktop\ANPR_project\data.yaml")

    # Check if the file exists before passing it to YOLO
    if not data_yaml_path.exists():
        print(f"\nERROR: Cannot find data.yaml at: {data_yaml_path}")
        print("Please check your file tree in VS Code and verify where 'data.yaml' is located.\n")
        return

    print(f" Found dataset config at: {data_yaml_path}")
    print("Starting GPU Training on NVIDIA RTX 2050...")

    # Load base weights
    model = YOLO('yolov8n.pt')
    
    # Train on GPU
    results = model.train(
        data=str(data_yaml_path),
        epochs=80,
        imgsz=640,
        device=0, 
        amp=False,           
        project='GPU_Results',
        name='training_run_FP32'
    )

if __name__ == '__main__':
    main()