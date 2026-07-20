import os
import random
import shutil

# Define paths
train_img_dir = 'dataset/train/images'
train_lbl_dir = 'dataset/train/labels'
val_img_dir = 'dataset/val/images'
val_lbl_dir = 'dataset/val/labels'

# Create validation directories if they don't exist
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

# Get all images currently in the train folder
images = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Define split percentage (e.g., 20% for validation)
split_ratio = 0.20
num_val_images = int(len(images) * split_ratio)

# Randomly select images to move to the validation set
val_images = random.sample(images, num_val_images)

print(f"Total images found: {len(images)}")
print(f"Moving {num_val_images} image-label pairs to the validation folder...")

moved_count = 0

for img_name in val_images:
    # Construct base file name without extension
    base_name = os.path.splitext(img_name)[0]
    lbl_name = f"{base_name}.txt"
    
    # Define source paths
    src_img = os.path.join(train_img_dir, img_name)
    src_lbl = os.path.join(train_lbl_dir, lbl_name)
    
    # Define destination paths
    dst_img = os.path.join(val_img_dir, img_name)
    dst_lbl = os.path.join(val_lbl_dir, lbl_name)
    
    # Move image if it exists
    if os.path.exists(src_img):
        shutil.move(src_img, dst_img)
        
        # Move corresponding label file if it exists
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)
            moved_count += 1
        else:
            print(f"Warning: Missing label file for {img_name}")

print(f"Successfully moved {moved_count} pairs to validation!")