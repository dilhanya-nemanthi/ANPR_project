#splitting script for the first time

import os
import shutil
import random

# --- CONFIGURATION ---
SOURCE_IMAGES = r"C:\Users\User\Desktop\ANPR_project\VEHICLE_IMAGES\images"   # Path to your raw images folder
SOURCE_LABELS = r"C:\Users\User\Desktop\ANPR_project\VEHICLE_IMAGES\labels"   # Path to your raw labels folder
OUTPUT_DIR = r"dataset"      # Destination folder
TRAIN_RATIO = 0.8           # 80% train, 20% validation
# ---------------------

# 1. Create target directories
for split in ['train', 'val']:
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'), exist_ok=True)

# 2. Get list of valid images
valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
all_images = [f for f in os.listdir(SOURCE_IMAGES) if f.lower().endswith(valid_extensions)]

# Shuffle for random distribution
random.seed(42)  # Fixed seed for reproducibility
random.shuffle(all_images)

# 3. Calculate split index
split_idx = int(len(all_images) * TRAIN_RATIO)
train_files = all_images[:split_idx]
val_files = all_images[split_idx:]

print(f"Total paired images found: {len(all_images)}")
print(f"Splitting: {len(train_files)} -> Train | {len(val_files)} -> Validation\n")

def move_pairs(file_list, split_name):
    for img_file in file_list:
        base_name, _ = os.path.splitext(img_file)
        lbl_file = f"{base_name}.txt"

        src_img = os.path.join(SOURCE_IMAGES, img_file)
        src_lbl = os.path.join(SOURCE_LABELS, lbl_file)

        dst_img = os.path.join(OUTPUT_DIR, split_name, 'images', img_file)
        dst_lbl = os.path.join(OUTPUT_DIR, split_name, 'labels', lbl_file)

        # Copy image
        shutil.copy(src_img, dst_img)

        # Copy label if it exists, or create empty txt for background images
        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, dst_lbl)
        else:
            with open(dst_lbl, 'w') as f:
                pass  # Empty file for background images

move_pairs(train_files, 'train')
move_pairs(val_files, 'val')

print(" Dataset successfully created and organized!")