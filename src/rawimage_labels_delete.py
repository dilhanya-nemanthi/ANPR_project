import os

# --- CONFIGURATION ---
# Folder containing all your raw images
IMAGE_DIR = r"C:\Users\User\Desktop\ANPR_project\NEW-images\images" 

# Folder containing your 880 .txt label files
LABEL_DIR = r"C:\Users\User\Desktop\ANPR_project\NEW-images\labels" 

def clean_dataset():
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
    
    print("[*] Scanning folders...")
    
    # 1. Get raw lists of files
    label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.txt')]
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_exts)]
    
    # 2. Extract base names into sets for fast matching
    label_bases = {os.path.splitext(f)[0] for f in label_files}
    image_bases = {os.path.splitext(f)[0] for f in image_files}
    
    print(f"[*] Found {len(image_files)} total images and {len(label_files)} label files.")
    
    deleted_images = 0
    deleted_labels = 0
    
    # 3. Delete images without matching labels
    for img_name in image_files:
        base_name = os.path.splitext(img_name)[0]
        if base_name not in label_bases:
            img_path = os.path.join(IMAGE_DIR, img_name)
            os.remove(img_path)
            deleted_images += 1
            
    # 4. Delete labels without matching images
    for lbl_name in label_files:
        base_name = os.path.splitext(lbl_name)[0]
        if base_name not in image_bases:
            lbl_path = os.path.join(LABEL_DIR, lbl_name)
            os.remove(lbl_path)
            deleted_labels += 1
            
    print("-" * 40)
    print("Cleanup Complete!")
    print(f"Deleted {deleted_images} images that had no labels.")
    print(f"Deleted {deleted_labels} labels that had no images.")
    print(f"Images remaining: {len(image_files) - deleted_images}")
    print(f"Labels remaining: {len(label_files) - deleted_labels}")
    print("-" * 40)

if __name__ == "__main__":
    clean_dataset()