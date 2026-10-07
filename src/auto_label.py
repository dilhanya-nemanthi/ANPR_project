import os
import csv
import easyocr

# --- Configuration ---
IMG_DIR = r"finalized_plates"
CSV_OUTPUT = r"unified_labels.csv"

def main():
    print("[*] Initializing EasyOCR model...")
    # Initialize the reader for English characters, utilizing the GPU
    reader = easyocr.Reader(['en'], gpu=True) 

    print(f"[*] Scanning mixed images in {IMG_DIR}...")
    
    with open(CSV_OUTPUT, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        count = 0
        
        for filename in os.listdir(IMG_DIR):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(IMG_DIR, filename)
                
                # Perform OCR, restricting output to valid plate characters
                results = reader.readtext(img_path, detail=0, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                
                if results:
                    # Strip whitespace and ensure alphanumeric integrity
                    raw_text = "".join(results).upper().replace(" ", "")
                    clean_text = "".join(c for c in raw_text if c.isalnum())
                    
                    writer.writerow([filename, clean_text])
                    count += 1
                else:
                    # Mark failures for quick manual correction later
                    writer.writerow([filename, "REVIEW_NEEDED"])
                    
                if count % 500 == 0:
                    print(f"    -> Labeled {count} images...")

    print(f"\n[*] Auto-labeling complete! CSV saved to {CSV_OUTPUT}")

if __name__ == "__main__":
    main()