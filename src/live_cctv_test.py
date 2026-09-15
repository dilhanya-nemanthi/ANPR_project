import cv2
import threading
import queue
import time
import os
import csv
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from ultralytics import YOLO
from wpodnet import Predictor, load_wpodnet_from_checkpoint

# --- 1. CONFIGURATION & CSV SETUP ---
CSV_FILE = "detected_plates.csv"
COOLDOWN_SECONDS = 10.0  # Prevent logging duplicate plates within 10 seconds
recent_plates = {}       # In-memory dictionary: {plate_text: last_logged_timestamp}
csv_lock = threading.Lock()

def init_csv():
    """Create the CSV file with headers if it does not already exist."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Plate Text", "Confidence (%)"])
        print(f"[*] Initialized log file: {CSV_FILE}")

def log_plate_to_csv(plate_text, confidence):
    """Logs detected plates with timestamp while filtering out noise and duplicates."""
    # Filter out empty, unknown, or noisy 1-2 character misreads
    if not plate_text or plate_text == "UNKNOWN" or len(plate_text) < 3:
        return

    current_time = time.time()
    last_logged_time = recent_plates.get(plate_text, 0)

    # Check cooldown threshold
    if current_time - last_logged_time > COOLDOWN_SECONDS:
        recent_plates[plate_text] = current_time
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with csv_lock:
            with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp_str, plate_text, f"{confidence:.1f}"])

        print(f"\n[CSV LOGGED] {timestamp_str} | Plate: {plate_text} | Conf: {confidence:.1f}%\n")

# --- 2. OCR ARCHITECTURE & SETUP ---
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

# --- 3. THREAD QUEUES ---
frame_queue = queue.Queue(maxsize=5)
display_queue = queue.Queue(maxsize=5)
running = True 

def capture_frames(source):
    """THREAD 1: Network Capture (Producer)"""
    global running
    print("[THREAD 1] Connecting to stream...")
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[THREAD 1] ERROR: Could not open the stream.")
        running = False
        return

    while running:
        ret, frame = cap.read()
        if ret:
            if frame_queue.full():
                frame_queue.get() 
            frame_queue.put(frame)
        else:
            time.sleep(0.1)

    cap.release()
    print("[THREAD 1] Capture stopped.")

def process_detection(yolo_path, wpod_path, ocr_path):
    """THREAD 2: Three-Stage Detection & CSV Logging"""
    global running
    print("[THREAD 2] Loading AI models...")
    
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

            # --- STAGE 1: YOLO Rough Bounding Box ---
            yolo_results = yolo_model.predict(source=frame, conf=0.5, verbose=False)
            
            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                yolo_conf = float(box.conf[0]) * 100
                h, w = frame.shape[:2]
                
                # --- STAGE 2: WPOD-NET Context Search ---
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                box_w, box_h = x2 - x1, y2 - y1
                crop_size = int(max(box_w, box_h) * 2.0)
                half_size = crop_size // 2
                
                cy1, cy2 = max(0, cy - half_size), min(h, cy + half_size)
                cx1, cx2 = max(0, cx - half_size), min(w, cx + half_size)
                
                plate_crop = frame[cy1:cy2, cx1:cx2]
                if plate_crop.size == 0: continue
                
                rgb_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_crop)
                
                predicted_text = "UNKNOWN"
                final_conf = yolo_conf
                
                try:
                    prediction = wpod_predictor.predict(pil_image, scaling_ratio=1.0)
                    
                    if prediction.confidence >= 0.5:
                        final_conf = prediction.confidence * 100
                        
                        # Map WPOD-NET's 4 corners back to the full frame
                        pts_full = []
                        for px, py in prediction.bounds:
                            pts_full.append([cx1 + px, cy1 + py])
                        pts_full_np = np.array(pts_full, dtype=np.float32)
                        
                        # --- STAGE 3: FLATTEN THE PLATE ---
                        # Map the angled corners to a perfectly flat 128x32 rectangle
                        dst_pts = np.array([[0, 0], [128, 0], [128, 32], [0, 32]], dtype=np.float32)
                        matrix = cv2.getPerspectiveTransform(pts_full_np, dst_pts)
                        flat_plate = cv2.warpPerspective(frame, matrix, (128, 32))
                        
                        # --- STAGE 4: CRNN OCR ON THE FLAT PLATE ---
                        gray_crop = cv2.cvtColor(flat_plate, cv2.COLOR_BGR2GRAY)
                        normalized = (gray_crop / 255.0 - 0.5) / 0.5
                        tensor_crop = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                        with torch.no_grad():
                            preds = ocr_model(tensor_crop)
                        predicted_text = decode_predictions(preds)

                        # Draw the WPOD angled polygon and the accurate text
                        pts_draw = np.array(pts_full, dtype=np.int32)
                        cv2.polylines(annotated_frame, [pts_draw], isClosed=True, color=(0, 255, 0), thickness=3)
                        
                        label = f"{predicted_text} ({final_conf:.1f}%)"
                        text_pos = (pts_draw[0][0], max(25, pts_draw[0][1] - 10))
                        cv2.putText(annotated_frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                        cv2.putText(annotated_frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                    else:
                        raise Exception("WPOD low confidence")
                        
                except Exception:
                    # FALLBACK: If WPOD fails, OCR the raw YOLO box anyway
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    raw_crop = frame[y1:y2, x1:x2]
                    
                    if raw_crop.size > 0:
                        gray_crop = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
                        gray_crop = cv2.resize(gray_crop, (128, 32))
                        normalized = (gray_crop / 255.0 - 0.5) / 0.5
                        tensor_crop = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            preds = ocr_model(tensor_crop)
                        predicted_text = decode_predictions(preds)
                        
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{predicted_text} ({yolo_conf:.1f}%)"
                    cv2.putText(annotated_frame, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Save to CSV
                log_plate_to_csv(predicted_text, final_conf)

            if display_queue.full():
                display_queue.get()
            display_queue.put(annotated_frame)
        else:
            time.sleep(0.01)
def main():
    """MAIN THREAD: UI Display (Consumer)"""
    global running
    print("--- Starting Multi-Threaded ANPR Pipeline ---")

    # Initialize CSV log file
    init_csv()

    rtsp_url = "rtsp://admin:It%40123aasl@10.64.64.16/Streaming/channels/001/?transportmode=unicast"
    
    # Path configurations
    yolo_path = r"runs\detect\runs\detect\yolov8n_augmented_scratch-3\weights\best.pt"
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
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[MAIN] Shutting down...")
            running = False
            break

    cap_thread.join()
    det_thread.join()
    cv2.destroyAllWindows()
    print("[MAIN] System offline. All records saved to", CSV_FILE)

if __name__ == "__main__":
    main()