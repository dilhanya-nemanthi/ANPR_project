import cv2
import threading
import queue
import time
import os
import csv
from datetime import datetime
from collections import deque, Counter
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from ultralytics import YOLO
from wpodnet import Predictor, load_wpodnet_from_checkpoint
import uuid

# Force FFmpeg to minimize RTSP network buffer delay
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000"

# --- 1. CONFIGURATION & CSV SETUP ---
EDGE_CASE_DIR = "needs_review"
os.makedirs(EDGE_CASE_DIR, exist_ok=True)

CSV_FILE = "detected_plates.csv"
COOLDOWN_SECONDS = 10.0  
recent_plates = {}       
csv_lock = threading.Lock()

# --- NEW: TRIPWIRE CONFIGURATION ---
# Set the line at 70% of the screen height. 
# Adjust to 0.5 for the middle, or 0.85 for closer to the bottom.
TRIPWIRE_Y_RATIO = 0.70 

# Temporal smoothing buffers
plate_history = deque(maxlen=8)
STABILITY_THRESHOLD = 4  

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Plate Text", "Confidence (%)"])
        print(f"[*] Initialized log file: {CSV_FILE}")

def log_plate_to_csv(plate_text, confidence):
    if not plate_text or plate_text in ("UNKNOWN", "ANALYZING...", "REJECTED") or len(plate_text) < 3:
        return

    current_time = time.time()
    last_logged_time = recent_plates.get(plate_text, 0)

    if current_time - last_logged_time > COOLDOWN_SECONDS:
        recent_plates[plate_text] = current_time
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with csv_lock:
            with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp_str, plate_text, f"{confidence:.1f}"])

        print(f"\n[CSV LOGGED] {timestamp_str} | Plate: {plate_text} | Conf: {confidence:.1f}%\n")

def harvest_edge_cases(crop_img, confidence, text_prediction):
    if 40.0 <= confidence <= 85.0 and crop_img is not None and crop_img.size > 0:
        unique_id = uuid.uuid4().hex[:8]
        safe_text = "".join(c for c in text_prediction if c.isalnum()) or "UNKNOWN"
        filename = f"edge_{safe_text}_{confidence:.0f}_{unique_id}.jpg"
        filepath = os.path.join(EDGE_CASE_DIR, filename)
        
        cv2.imwrite(filepath, crop_img)
        print(f"[ACTIVE LEARNING] Saved edge case: {filepath}")

# --- 2. POST-PROCESSING & STABILIZATION ---
def correct_plate_syntax(raw_text):
    if not raw_text or len(raw_text) < 5 or len(raw_text) > 9:
        return "REJECTED"

    letter_to_num = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'D': '0'}
    chars = list(raw_text)

    num_suffix_len = min(4, len(chars))
    for i in range(len(chars) - num_suffix_len, len(chars)):
        if chars[i] in letter_to_num:
            chars[i] = letter_to_num[chars[i]]

    return "".join(chars)

def stabilize_prediction(candidate_text):
    if not candidate_text or candidate_text in ("UNKNOWN", "REJECTED"):
        return "ANALYZING..."

    plate_history.append(candidate_text)
    most_common, count = Counter(plate_history).most_common(1)[0]

    if count >= STABILITY_THRESHOLD:
        return most_common
    return candidate_text

# --- 3. OCR ARCHITECTURE & SETUP ---
VOCAB = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
INT_TO_CHAR = {idx + 1: char for idx, char in enumerate(VOCAB)}
NUM_CLASSES = len(VOCAB) + 1

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

# --- 4. THREAD QUEUES ---
frame_queue = queue.Queue(maxsize=3)
display_queue = queue.Queue(maxsize=3)
running = True 

def capture_frames(source):
    global running
    print("[THREAD 1] Connecting to RTSP stream...")
    
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[THREAD 1] ERROR: Could not open the RTSP stream.")
        running = False
        return

    while running:
        ret, frame = cap.read()
        if ret:
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame)
        else:
            time.sleep(0.05)

    cap.release()
    print("[THREAD 1] Capture stopped.")

def process_detection(yolo_path, wpod_path, ocr_path):
    global running
    print("[THREAD 2] Loading AI models on CUDA/GPU...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    yolo_model = YOLO(yolo_path)
    wpod_model = load_wpodnet_from_checkpoint(wpod_path).to(device)
    wpod_predictor = Predictor(wpod_model)
    
    ocr_model = CRNN(NUM_CLASSES).to(device)
    ocr_model.load_state_dict(torch.load(ocr_path, map_location=device))
    ocr_model.eval()
    
    while running:
        if not frame_queue.empty():
            frame = frame_queue.get()
            annotated_frame = frame.copy()
            
            h, w = frame.shape[:2]
            
            # --- NEW: Draw the default Tripwire (Orange) ---
            tripwire_y = int(h * TRIPWIRE_Y_RATIO)
            cv2.line(annotated_frame, (0, tripwire_y), (w, tripwire_y), (0, 165, 255), 2)

            yolo_results = yolo_model.predict(source=frame, conf=0.70, verbose=False)
            
            for box in yolo_results[0].boxes:
                raw_x1, raw_y1, raw_x2, raw_y2 = map(int, box.xyxy[0])
                yolo_conf = float(box.conf[0]) * 100
                
                box_w = raw_x2 - raw_x1
                box_h = raw_y2 - raw_y1
                
                if box_h == 0: 
                    continue
                    
                aspect_ratio = box_w / float(box_h)
                
                if aspect_ratio < 1.2 or aspect_ratio > 5.5:
                    continue 

                pad = 10
                x1 = max(0, raw_x1 - pad)
                y1 = max(0, raw_y1 - pad)
                x2 = min(w, raw_x2 + pad)
                y2 = min(h, raw_y2 + pad)
                
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                padded_w = x2 - x1
                padded_h = y2 - y1
                crop_size = int(max(padded_w, padded_h) * 2.0)
                half_size = crop_size // 2
                
                cy1, cy2 = max(0, cy - half_size), min(h, cy + half_size)
                cx1, cx2 = max(0, cx - half_size), min(w, cx + half_size)
                
                plate_crop = frame[cy1:cy2, cx1:cx2]
                if plate_crop.size == 0:
                    continue
                
                rgb_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_crop)
                
                candidate_text = "UNKNOWN"
                final_conf = yolo_conf
                harvest_crop = None
                
                try:
                    prediction = wpod_predictor.predict(pil_image, scaling_ratio=1.0)
                    
                    if prediction.confidence >= 0.5:
                        final_conf = prediction.confidence * 100
                        
                        pts_full = []
                        for px, py in prediction.bounds:
                            pts_full.append([cx1 + px, cy1 + py])
                        pts_full_np = np.array(pts_full, dtype=np.float32)
                        
                        dst_pts = np.array([[0, 0], [128, 0], [128, 32], [0, 32]], dtype=np.float32)
                        matrix = cv2.getPerspectiveTransform(pts_full_np, dst_pts)
                        flat_plate = cv2.warpPerspective(frame, matrix, (128, 32))
                        harvest_crop = flat_plate.copy()
                        
                        gray_crop = cv2.cvtColor(flat_plate, cv2.COLOR_BGR2GRAY)
                        normalized = (gray_crop / 255.0 - 0.5) / 0.5
                        tensor_crop = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                        with torch.no_grad():
                            preds = ocr_model(tensor_crop)
                        candidate_text = decode_predictions(preds)

                        pts_draw = np.array(pts_full, dtype=np.int32)
                        cv2.polylines(annotated_frame, [pts_draw], isClosed=True, color=(0, 255, 0), thickness=3)
                        text_pos = (pts_draw[0][0], max(25, pts_draw[0][1] - 10))
                    else:
                        raise Exception("WPOD low confidence")
                        
                except Exception:
                    raw_crop = frame[y1:y2, x1:x2]
                    
                    if raw_crop.size > 0:
                        harvest_crop = raw_crop.copy()
                        gray_crop = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
                        gray_crop = cv2.resize(gray_crop, (128, 32))
                        normalized = (gray_crop / 255.0 - 0.5) / 0.5
                        tensor_crop = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            preds = ocr_model(tensor_crop)
                        candidate_text = decode_predictions(preds)
                        
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text_pos = (x1, max(25, y1 - 10))

                harvest_edge_cases(harvest_crop, final_conf, candidate_text)
                corrected_text = correct_plate_syntax(candidate_text)
                
                if corrected_text == "REJECTED":
                    label = f"REJECTED: {candidate_text} ({final_conf:.1f}%)"
                    cv2.putText(annotated_frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                    cv2.putText(annotated_frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    stable_text = stabilize_prediction(corrected_text)
                    label = f"{stable_text} ({final_conf:.1f}%)"
                    cv2.putText(annotated_frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                    cv2.putText(annotated_frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # --- NEW: Tripwire Trigger Logic ---
                    # Only save to CSV if the center of the plate (cy) has crossed the tripwire (tripwire_y)
                    # Assumes cars drive *towards* the camera (moving from top to bottom)
                    if cy > tripwire_y:
                        # Briefly flash the line GREEN to show a successful crossing was logged
                        cv2.line(annotated_frame, (0, tripwire_y), (w, tripwire_y), (0, 255, 0), 4)
                        log_plate_to_csv(stable_text, final_conf)

            if display_queue.full():
                try:
                    display_queue.get_nowait()
                except queue.Empty:
                    pass
            display_queue.put(annotated_frame)
        else:
            time.sleep(0.005)

def main():
    global running
    print("--- Starting Multi-Threaded ANPR Pipeline ---")

    init_csv()

    rtsp_url = "rtsp://admin:It%40123aasl@10.64.64.16/Streaming/channels/001/?transportmode=unicast"
    
    yolo_path = r"runs\detect\runs\detect\yolov8n_augmented_scratch-5\weights\best.pt"
    wpod_path = r"weights\wpodnet.pth"
    ocr_path = r"weights\crnn_ocr_best.pt"

    cap_thread = threading.Thread(target=capture_frames, args=(rtsp_url,), daemon=True)
    cap_thread.start()

    det_thread = threading.Thread(
        target=process_detection, 
        args=(yolo_path, wpod_path, ocr_path), 
        daemon=True
    )
    det_thread.start()

    print("[MAIN] Display feed initialized. Press 'q' to quit.")

    while running:
        if not display_queue.empty():
            final_frame = display_queue.get()
            cv2.imshow("Live RTSP ANPR Pipeline", final_frame)
        else:
            time.sleep(0.005)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[MAIN] Shutting down...")
            running = False
            break

    cap_thread.join(timeout=1.0)
    det_thread.join(timeout=1.0)
    cv2.destroyAllWindows()
    print("[MAIN] System offline. All records saved to", CSV_FILE)

if __name__ == "__main__":
    main()