import os

# --- CONFIGURATION ---
# Folder containing all your raw images
IMAGE_DIR = r"C:\Users\User\Desktop\ANPR_project\NEW-images\images" 

# Folder containing your 880 .txt label files
LABEL_DIR = r"C:\Users\User\Desktop\ANPR_project\NEW-images\labels" 

def clean_unlabeled_images():
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
    
    print("[*] Scanning folders...")
    
    # 1. Get a set of all label filenames (without the .txt extension)
    label_files = [os.path.splitext(f)[0] for f in os.listdir(LABEL_DIR) if f.endswith('.txt')]
    label_set = set(label_files)
    
    # 2. Get a list of all images
    images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_exts)]
    print(f"[*] Found {len(images)} total images and {len(label_set)} label files.")
    
    deleted_count = 0
    
    # 3. Check each image. If no matching label exists, delete it.
    for img_name in images:
        base_name = os.path.splitext(img_name)[0]
        
        if base_name not in label_set:
            img_path = os.path.join(IMAGE_DIR, img_name)
            os.remove(img_path)
            deleted_count += 1
            
    print("-" * 40)
    print("Cleanup Complete!")
    print(f"Deleted {deleted_count} images that had no labels.")
    print(f"Images remaining: {len(images) - deleted_count}")
    print("-" * 40)

if __name__ == "__main__":
    clean_unlabeled_images()