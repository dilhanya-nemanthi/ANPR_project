import os
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = r"runs\detect\runs\detect\yolov8n_augmented_scratch-3\weights\best.pt"
UNLABELED_IMAGES_DIR = r"C:\Users\User\Downloads\vehicles_3354\vehicles_3354"
OUTPUT_LABELS_DIR = r"C:\Users\User\Desktop\NEW-images\newlbls" 

os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

def auto_annotate():
    print("[*] Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
    image_files = [f for f in os.listdir(UNLABELED_IMAGES_DIR) if f.lower().endswith(valid_exts)]
    
    print(f"[*] Found {len(image_files)} images. Starting auto-annotation...")
    
    success_count = 0
    background_count = 0
    
    for img_name in image_files:
        img_path = os.path.join(UNLABELED_IMAGES_DIR, img_name)
        
        # Run inference
        results = model.predict(source=img_path, conf=0.7, verbose=False)
        
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(OUTPUT_LABELS_DIR, txt_name)
        
        boxes = results[0].boxes
        
        # If a plate is found, write the coordinates
        if len(boxes) > 0:
            with open(txt_path, 'w') as f:
                for box in boxes:
                    cls = int(box.cls[0])
                    x_c, y_c, w, h = box.xywhn[0].tolist()
                    f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            success_count += 1
        
        # If NO plate is found, create an EMPTY .txt file
        else:
            open(txt_path, 'w').close()
            background_count += 1

    print("-" * 40)
    print("Auto-Annotation Complete!")
    print(f"Plates found: {success_count}")
    print(f"Background images (empty labels): {background_count}")
    print(f"Total labels generated: {success_count + background_count}")
    print(f"Labels saved to: {OUTPUT_LABELS_DIR}")
    print("-" * 40)

def delete_empty_labels():
    print("[*] Scanning for empty label files...")
    empty_count = 0
    
    for f in os.listdir(OUTPUT_LABELS_DIR):
        file_path = os.path.join(OUTPUT_LABELS_DIR, f)
        
        # os.path.getsize() returns the size of the file to check if it is 0 bytes
        if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
            # os.remove() deletes the file path from the system
            os.remove(file_path)
            empty_count += 1
            
    print(f"[*] Cleanup Complete: Deleted {empty_count} empty label files.")

if __name__ == "__main__":
    #auto_annotate()
    # Call the new function immediately after annotation finishes
    delete_empty_labels()