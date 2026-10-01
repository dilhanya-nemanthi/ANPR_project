import os
import random
import csv
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---
NUM_PLATES = 10000
OUTPUT_DIR = "synthetic_plates"
CSV_FILE = "synthetic_labels.csv"
FONT_PATH = "assets/CharlesWright-Bold.otf" # Updated to your OTF font

os.makedirs(OUTPUT_DIR, exist_ok=True)
vocab_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
vocab_numbers = "0123456789"

def generate_plate_string():
    """Generates random valid Sri Lankan plate formats (e.g., CBA8234, WPCAQ1234)"""
    num_letters = random.choice([2, 3])
    letters = "".join(random.choices(vocab_letters, k=num_letters))
    numbers = "".join(random.choices(vocab_numbers, k=4))
    
    if random.random() > 0.5:
        region = random.choice(["WP", "CP", "NW", "SP", "EP", "NC"])
        return f"{region}{letters}{numbers}"
    return f"{letters}{numbers}"

def create_blank_plate(width=520, height=110):
    """Dynamically generates a yellow or white license plate background."""
    # 50/50 chance for Yellow or White plate
    if random.random() > 0.5:
        bg_color = (255, 204, 0) # Standard license plate yellow
    else:
        bg_color = (240, 240, 240) # Off-white (more realistic than pure 255 white)
        
    border_color = (0, 0, 0) # Black border
    
    # Create solid color image
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Add a 4-pixel black border
    draw.rectangle([(0, 0), (width-1, height-1)], outline=border_color, width=4)
    return img, draw

def add_cctv_noise(cv_img):
    """Applies artificial degradation to bridge the domain gap"""
    # 1. Random Motion Blur
    if random.random() > 0.5:
        kernel_size = random.choice([(3,3), (5,5)])
        cv_img = cv2.blur(cv_img, kernel_size)
        
    # 2. Gaussian Noise (CCTV static)
    if random.random() > 0.3:
        row, col, ch = cv_img.shape
        mean = 0
        sigma = random.randint(10, 30) 
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        cv_img = cv2.add(cv_img, gauss.astype('uint8'))
        
    # 3. Random contrast shift
    alpha = random.uniform(0.7, 1.3)
    cv_img = cv2.convertScaleAbs(cv_img, alpha=alpha, beta=0)
    
    return cv_img

# --- Generation Loop ---
with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    
    try:
        # 80 is a good scale for a 520x110 plate, adjust if the font looks too large/small
        font = ImageFont.truetype(FONT_PATH, 80) 
    except IOError:
        print(f"[!] ERROR: Font file not found at {FONT_PATH}.")
        exit()

    print(f"[*] Generating {NUM_PLATES} synthetic plates...")
    
    for i in range(NUM_PLATES):
        plate_text = generate_plate_string()
        filename = f"synth_{i:05d}.jpg"
        
        # Dynamically create the background (now randomly yellow or white)
        base_img, draw = create_blank_plate()
        
        # Calculate text position to center it
        bbox = draw.textbbox((0, 0), plate_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        img_w, img_h = base_img.size
        x = (img_w - text_w) / 2
        
        # Slightly adjust Y to visually center the Charles Wright font
        y = (img_h - text_h) / 2 - 10 
        
        # Draw text in black
        draw.text((x, y), plate_text, font=font, fill=(0, 0, 0))
        
        # Convert to OpenCV format to apply CCTV degradation
        cv_img = cv2.cvtColor(np.array(base_img), cv2.COLOR_RGB2BGR)
        noisy_img = add_cctv_noise(cv_img)
        
        # Resize to your CRNN's required dimensions (128x32)
        final_img = cv2.resize(noisy_img, (128, 32))
        
        # Save image and write to CSV
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), final_img)
        writer.writerow([filename, plate_text])
        
        if i % 1000 == 0 and i > 0:
            print(f"    -> Generated {i} plates...")

print(f"[*] Done! Saved to '{OUTPUT_DIR}' and '{CSV_FILE}'")