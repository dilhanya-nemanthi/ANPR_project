import os
import random
import cv2
import numpy as np
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---
NUM_PLATES = 4000
OUTPUT_DIR = "synthetic_plates_only"
FONT_PATH = "assets/CharlesWright-Bold.otf"

os.makedirs(OUTPUT_DIR, exist_ok=True)
vocab_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
vocab_numbers = "0123456789"
used_filenames = set()

def generate_plate_string():
    num_letters = random.choice([2, 3])
    letters = "".join(random.choices(vocab_letters, k=num_letters))
    numbers = "".join(random.choices(vocab_numbers, k=4))
    
    if random.random() > 0.5:
        region = random.choice(["WP", "CP", "NW", "SP", "EP", "NC"])
        return f"{region}{letters}{numbers}"
    return f"{letters}{numbers}"

def create_blank_plate(width=520, height=110):
    if random.random() > 0.5:
        bg_color = (255, 204, 0) 
    else:
        bg_color = (240, 240, 240) 
        
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (width-1, height-1)], outline=(0, 0, 0), width=4)
    return img, draw

def add_cctv_noise(cv_img):
    if random.random() > 0.5:
        cv_img = cv2.blur(cv_img, random.choice([(3,3), (5,5)]))
        
    if random.random() > 0.3:
        row, col, ch = cv_img.shape
        gauss = np.random.normal(0, random.randint(10, 30), (row, col, ch)).reshape(row, col, ch)
        cv_img = cv2.add(cv_img, gauss.astype('uint8'))
        
    cv_img = cv2.convertScaleAbs(cv_img, alpha=random.uniform(0.7, 1.3), beta=0)
    return cv_img

def get_random_timestamp_filename():
    """Generates a unique timestamp format: 2023-09-03 00-01-21.jpg"""
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31, 23, 59, 59)
    
    while True:
        # Calculate random seconds within the 3-year window
        random_seconds = random.randint(0, int((end_date - start_date).total_seconds()))
        random_date = start_date + timedelta(seconds=random_seconds)
        
        # Format strictly to the requested CCTV style
        filename = random_date.strftime("%Y-%m-%d %H-%M-%S.jpg")
        
        # Ensure we don't accidentally overwrite an image with the exact same second
        if filename not in used_filenames:
            used_filenames.add(filename)
            return filename

# --- Generation Loop ---
try:
    font = ImageFont.truetype(FONT_PATH, 80) 
except IOError:
    print(f"[!] ERROR: Font file not found at {FONT_PATH}.")
    exit()

print(f"[*] Generating {NUM_PLATES} synthetic plates for academic dataset...")

for i in range(NUM_PLATES):
    plate_text = generate_plate_string()
    filename = get_random_timestamp_filename()
    
    base_img, draw = create_blank_plate()
    bbox = draw.textbbox((0, 0), plate_text, font=font)
    x = (base_img.size[0] - (bbox[2] - bbox[0])) / 2
    y = (base_img.size[1] - (bbox[3] - bbox[1])) / 2 - 10 
    
    draw.text((x, y), plate_text, font=font, fill=(0, 0, 0))
    
    cv_img = cv2.cvtColor(np.array(base_img), cv2.COLOR_RGB2BGR)
    noisy_img = add_cctv_noise(cv_img)
    final_img = cv2.resize(noisy_img, (128, 32))
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), final_img)
    
    if i % 1000 == 0 and i > 0:
        print(f"    -> Generated {i} plates...")

print(f"[*] Done! Saved strictly images to '{OUTPUT_DIR}'")