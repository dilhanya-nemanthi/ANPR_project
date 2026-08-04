import os
import random
import shutil
from pathlib import Path

def main():
    print("--- Starting Data Split ---")

    # 1. Define the source folders
    new_images_dir = Path("New images/images")
    new_labels_dir = Path("New images/labels")
    
    # 2. Define the destination folders
    train_img_dir = Path("dataset/train/images")
    train_lbl_dir = Path("dataset/train/labels")
    val_img_dir = Path("dataset/val/images")
    val_lbl_dir = Path("dataset/val/labels")

    # NEW: Automatically create the destination folders if they don't exist
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Verify source folders exist
    if not new_images_dir.exists() or not new_labels_dir.exists():
        print("ERROR: Cannot find the 'New images/images' or 'New images/labels' folders.")
        return

    # 3. Get all images and shuffle them randomly
    images = [f for f in os.listdir(new_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)

    # 4. Calculate the 80/20 split
    split_idx = int(len(images) * 0.8)
    train_files = images[:split_idx]
    val_files = images[split_idx:]

    print(f"Found {len(images)} new images.")
    print(f"Allocating {len(train_files)} to Training and {len(val_files)} to Validation...\n")

    # 5. Function to copy files
    def copy_files(file_list, dest_img, dest_lbl):
        for img_name in file_list:
            src_img = new_images_dir / img_name
            lbl_name = os.path.splitext(img_name)[0] + ".txt"
            src_lbl = new_labels_dir / lbl_name

            # Copy image
            if src_img.exists():
                shutil.copy(src_img, dest_img / img_name)
            
            # Copy label
            if src_lbl.exists():
                shutil.copy(src_lbl, dest_lbl / lbl_name)
            else:
                print(f" WARNING: No matching label file found for {img_name}")

    # 6. Execute the copying
    copy_files(train_files, train_img_dir, train_lbl_dir)
    copy_files(val_files, val_img_dir, val_lbl_dir)
    
    print("\n Success! All new files have been merged into your dataset.")

if __name__ == "__main__":
    main()