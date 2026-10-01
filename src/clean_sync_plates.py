import os
import pandas as pd

# Update these paths if your script is located outside the main ANPR folder
CSV_FILE = r"C:\Users\User\Desktop\ANPR_project\Number plates\plates-labels.csv"
IMAGE_DIR = r"C:\Users\User\Desktop\ANPR_project\Number plates\plates-images"

# 1. Read the CSV and extract all valid filenames into a fast lookup set
# Assuming column 0 contains the filename (e.g., 'car_1.jpg')
df = pd.read_csv(CSV_FILE, header=None)
labeled_filenames = set(df[0].astype(str))

print(f"[*] Found {len(labeled_filenames)} labeled images in CSV.")

deleted_count = 0

# 2. Iterate through every actual file in the plates-images folder
for img_file in os.listdir(IMAGE_DIR):
    # 3. If the image is NOT in the CSV, delete it from the hard drive
    if img_file not in labeled_filenames:
        file_path = os.path.join(IMAGE_DIR, img_file)
        try:
            os.remove(file_path)
            deleted_count += 1
            print(f"[Deleted] {img_file}")
        except Exception as e:
            print(f"[Error] Could not delete {img_file}: {e}")

print(f"\n[*] Cleanup complete. Successfully deleted {deleted_count} unlabeled images.")