import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO

# --- OCR SETUP ---
# Ensure this points to your trained CRNN weights!
OCR_WEIGHTS = r"weights\crnn_ocr_best.pt" 
VOCAB = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
INT_TO_CHAR = {idx + 1: char for idx, char in enumerate(VOCAB)}
NUM_CLASSES = len(VOCAB) + 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- OCR MODEL ARCHITECTURE ---
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d((2, 1))
        )
        self.rnn = nn.GRU(256 * 2, 128, bidirectional=True, batch_first=True, num_layers=2)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        b, c, h, w = features.size()
        features = features.permute(0, 3, 1, 2).reshape(b, w, c * h)
        rnn_out, _ = self.rnn(features)
        logits = self.fc(rnn_out)
        return logits.permute(1, 0, 2).log_softmax(2)

def decode_predictions(preds):
    _, max_indices = torch.max(preds, 2)
    max_indices = max_indices.squeeze(1).cpu().numpy()
    
    decoded_text = []
    prev_char = -1
    for char_idx in max_indices:
        if char_idx != 0 and char_idx != prev_char:
            decoded_text.append(INT_TO_CHAR[char_idx])
        prev_char = char_idx
    return "".join(decoded_text)

# --- MAIN PIPELINE ---
def main():
    print("--- Script Started ---")
    
    # 1. Check if the models exist
    model_path = r"runs\detect\GPU_Results\training_run_FP32-2\weights\best.pt"
    if not os.path.exists(model_path):
        print(f"ERROR: Cannot find YOLO model at {model_path}")
        return
        
    if not os.path.exists(OCR_WEIGHTS):
        print(f"ERROR: Cannot find OCR model at {OCR_WEIGHTS}")
        return

    # 2. Check if the test image exists
    image_path = "img.jpg"  
    if not os.path.exists(image_path):
        print(f"ERROR: Cannot find image at {image_path}. Did you name it correctly?")
        return

    # 3. Load Models
    print("Model and image found! Loading YOLO...")
    model = YOLO(model_path)
    
    print("Loading CRNN OCR Model...")
    ocr_model = CRNN(NUM_CLASSES).to(DEVICE)
    ocr_model.load_state_dict(torch.load(OCR_WEIGHTS, map_location=DEVICE))
    ocr_model.eval()

    # 4. Read Image and Run YOLO
    print(f"Running detection on {image_path}...")
    img = cv2.imread(image_path)
    results = model.predict(source=img, save=False, conf=0.4) 
    
    annotated_img = img.copy()

    # 5. Process Detections & OCR
    for box_data in results[0].boxes:
        # Get coordinates
        box = box_data.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        
        # Crop the plate
        plate_crop = img[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue
            
        # OCR Preprocessing
        gray_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        gray_crop = cv2.resize(gray_crop, (128, 32))
        normalized = (gray_crop / 255.0 - 0.5) / 0.5
        tensor_crop = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # OCR Inference
        with torch.no_grad():
            preds = ocr_model(tensor_crop)
        predicted_text = decode_predictions(preds)
        
        print(f"SUCCESS: Detected Plate Text -> {predicted_text}")

        # Draw Bounding Box and Text
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (w, h), _ = cv2.getTextSize(predicted_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(annotated_img, (x1, y1 - 30), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(annotated_img, predicted_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # 6. Display Output
    print("Opening window...")
    cv2.imshow("License Plate Detection + OCR", annotated_img)
    print("Window opened! Click on the image window and press ANY KEY to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()