#USED TO LABEL CROPPED PLATES IMAGES WITH THEIR TEXTS AND SAVE THEM IN A CSV FILE
import os
import cv2

# Configuration
CROPS_FOLDER = r"C:\Users\User\Desktop\ANPR_project\Number plates\plates-images"        # Ensure this points to your cropped plates folder
OUTPUT_CSV = r"C:\Users\User\Desktop\ANPR_project\Number plates\plates-labels.csv"     # Output file path

valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
image_files = [f for f in os.listdir(CROPS_FOLDER) if f.lower().endswith(valid_exts)]

# Load existing labels so you can resume if you stop halfway
labeled = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if parts:
                labeled.add(parts[0])

print(f"Total images: {len(image_files)} | Already labeled: {len(labeled)}")
print("Instructions:")
print("- Type the plate text and hit ENTER.")
print("- Leave blank and hit ENTER to skip.")
print("- Type 'exit' or press Ctrl+C to save and quit.\n")

with open(OUTPUT_CSV, 'a') as f:
    try:
        for idx, img_name in enumerate(image_files, 1):
            if img_name in labeled:
                continue

            img_path = os.path.join(CROPS_FOLDER, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # Resize image for clear display
            display_img = cv2.resize(img, (400, 150), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Plate Preview", display_img)
            
            # --- THE FIX ---
            # Give OpenCV and Windows 100ms to physically draw the new image 
            # before the input() function freezes the script.
            cv2.waitKey(100) 

            user_input = input(f"[{idx}/{len(image_files)}] Text for '{img_name}': ").strip().upper()

            if user_input == "EXIT":
                print("\nSaving progress and exiting...")
                break

            if user_input:
                # Clean string (remove spaces and hyphens)
                clean_text = user_input.replace(" ", "").replace("-", "")
                f.write(f"{img_name},{clean_text}\n")
                f.flush()  # Save progress immediately

    except KeyboardInterrupt:
        print("\n\n[INFO] Script stopped manually with Ctrl+C. Your progress is saved!")
        
    finally:
        cv2.destroyAllWindows()
        print(f"Progress safely logged in {OUTPUT_CSV}")