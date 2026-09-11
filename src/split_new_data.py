#Splitting data for second time to merge new images with the existing dataset

import os
import shutil
import random

# --- CONFIGURATION ---
# Source directories
BASE_DIR = "VEHICLE_IMAGES"
SOURCE_IMAGES = os.path.join(BASE_DIR, "images")
SOURCE_LABELS = os.path.join(BASE_DIR, "labels")

# Output directory
OUTPUT_DIR = "dataset"

# Split ratio (20% for validation, 80% for training)
SPLIT_RATIO = 0.2 
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.webp')

def main():
    print(f"Scanning '{SOURCE_IMAGES}' for cleanup...")
    
    deleted_count = 0
    valid_images = []

    # --- STEP 1: CLEANUP ---
    for img_name in os.listdir(SOURCE_IMAGES):
        if not img_name.lower().endswith(VALID_EXTS):
            continue
            
        base_name = os.path.splitext(img_name)[0]
        txt_name = base_name + ".txt"
        
        img_path = os.path.join(SOURCE_IMAGES, img_name)
        label_path = os.path.join(SOURCE_LABELS, txt_name)
        
        # Check if label exists and has content
        if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
            os.remove(img_path)
            # If an empty text file exists, delete it too
            if os.path.exists(label_path):
                os.remove(label_path)
            deleted_count += 1
        else:
            valid_images.append(img_name)

    print(f"Deleted {deleted_count} unlabeled/empty images.")
    print(f"Found {len(valid_images)} perfectly matched pairs.")

    # --- STEP 2: SETUP FOLDERS ---
    print(f"\nCreating YOLO structure in '{OUTPUT_DIR}'...")
    dirs_to_make = [
        os.path.join(OUTPUT_DIR, 'images', 'train'),
        os.path.join(OUTPUT_DIR, 'images', 'val'),
        os.path.join(OUTPUT_DIR, 'labels', 'train'),
        os.path.join(OUTPUT_DIR, 'labels', 'val')
    ]
    
    # Remove existing dataset folder if it exists to ensure a clean slate
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        
    for d in dirs_to_make:
        os.makedirs(d, exist_ok=True)

    # --- STEP 3: SPLIT AND COPY ---
    # Shuffle the data
    random.seed(42)
    random.shuffle(valid_images)

    val_count = int(len(valid_images) * SPLIT_RATIO)
    val_images = valid_images[:val_count]
    train_images = valid_images[val_count:]

    def copy_files(image_list, split_type):
        for img_name in image_list:
            base_name = os.path.splitext(img_name)[0]
            txt_name = base_name + ".txt"

            # Source paths
            src_img = os.path.join(SOURCE_IMAGES, img_name)
            src_txt = os.path.join(SOURCE_LABELS, txt_name)
            
            # Destination paths
            dst_img = os.path.join(OUTPUT_DIR, 'images', split_type, img_name)
            dst_txt = os.path.join(OUTPUT_DIR, 'labels', split_type, txt_name)

            # Copy files
            shutil.copy(src_img, dst_img)
            shutil.copy(src_txt, dst_txt)

    print("Copying Training data...")
    copy_files(train_images, 'train')

    print("Copying Validation data...")
    copy_files(val_images, 'val')

    print("-" * 40)
    print("Dataset Preparation Complete!")
    print(f"Training pairs: {len(train_images)}")
    print(f"Validation pairs: {len(val_images)}")
    print("-" * 40)

if __name__ == "__main__":
    main()