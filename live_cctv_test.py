import cv2
import threading
import queue
import time
import os
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from wpodnet import Predictor, load_wpodnet_from_checkpoint # Your local PyTorch module

def load_wpod_net(json_path, h5_path):
    # 1. Read the JSON structure
    with open(json_path, 'r') as file:
        model_json = file.read()
        
    # 2. Build the empty model
    model = model_from_json(model_json)
    
    # 3. Inject the trained weights
    model.load_weights(h5_path)
    return model

# Create two thread-safe buffers
# frame_queue: raw images from camera -> AI
# display_queue: processed images from AI -> Screen
frame_queue = queue.Queue(maxsize=5)
display_queue = queue.Queue(maxsize=5)

running = True 

def capture_frames(source):
    """THREAD 1: Network Capture (Producer)"""
    global running
    print("[THREAD 1] Connecting to stream...")
    
    # Put the 0 directly into VideoCapture
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
 
def process_detection(yolo_path, wpod_path):
    """THREAD 2: Two-Stage Detection (YOLO Context -> PyTorch WPOD-NET)"""
    global running
    print("[THREAD 2] Loading AI models...")
    
    # 1. Load YOLO Model
    yolo_model = YOLO(yolo_path)
    
    # 2. Load PyTorch WPOD-NET Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    wpod_model = load_wpodnet_from_checkpoint(wpod_path).to(device)
    wpod_predictor = Predictor(wpod_model)
    
    while running:
        if not frame_queue.empty():
            # Pull raw frame
            frame = frame_queue.get()
            annotated_frame = frame.copy()

            # --- STAGE 1: YOLO Bounding Box ---
            yolo_results = yolo_model.predict(source=frame, conf=0.5, verbose=False)
            
            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                yolo_conf = float(box.conf[0]) * 100
                
                # --- MATHEMATICAL FIX: LARGE SQUARE CONTEXT CROP ---
                # 1. Find the exact center of the YOLO detection
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # 2. Force the crop to be a square that is 2x the width of the plate
                box_w = x2 - x1
                box_h = y2 - y1
                crop_size = int(max(box_w, box_h) * 2.0)
                half_size = crop_size // 2
                
                # 3. Calculate safe boundaries for the main frame
                h, w = frame.shape[:2]
                cy1 = max(0, cy - half_size)
                cy2 = min(h, cy + half_size)
                cx1 = max(0, cx - half_size)
                cx2 = min(w, cx + half_size)
                
                # Extract the generous square context from the frame
                plate_crop = frame[cy1:cy2, cx1:cx2]
                
                if plate_crop.size > 0:
                    rgb_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_crop)
                    
                    try:
                        # --- STAGE 2: WPOD-NET Corner & Accuracy Prediction ---
                        prediction = wpod_predictor.predict(pil_image, scaling_ratio=1.0)
                        
                        if prediction.confidence >= 0.5:
                            wpod_conf = prediction.confidence * 100
                            
                            # Map WPOD-NET's 4 angled corners back to the full main frame
                            pts_full = []
                            for px, py in prediction.bounds:
                                pts_full.append([int(cx1 + px), int(cy1 + py)])
                            pts_full = np.array(pts_full, dtype=np.int32)
                            
                            # Draw exact 4-corner polygon around the plate
                            cv2.polylines(annotated_frame, [pts_full], isClosed=True, color=(0, 255, 0), thickness=3)
                            
                            # Label with Name and Accuracy
                            label = f"Numberplate ({wpod_conf:.1f}%)"
                            text_pos = (pts_full[0][0], max(25, pts_full[0][1] - 10))
                            
                            # Text drop shadow for readability
                            cv2.putText(annotated_frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                            cv2.putText(annotated_frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            # Fallback: Draw YOLO box if WPOD confidence is low
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"Numberplate ({yolo_conf:.1f}%)"
                            cv2.putText(annotated_frame, label, (x1, max(25, y1 - 10)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                        
                    except Exception as e:
                        # Fallback: Draw YOLO box if WPOD-NET mathematics fail
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Numberplate ({yolo_conf:.1f}%)"
                        cv2.putText(annotated_frame, label, (x1, max(25, y1 - 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Push completed frame to UI queue
            if display_queue.full():
                display_queue.get()
            display_queue.put(annotated_frame)
        else:
            time.sleep(0.01)

    print("[THREAD 2] Detection stopped.")



def main():
    """MAIN THREAD: UI Display (Consumer)"""
    global running
    print("--- Starting Multi-Threaded Pipeline ---")

    rtsp_url = 'rtsp://admin:It%40123aasl@10.64.64.16/Streaming/channels/001/?transportmode=unicast'
    
    # 1. Define model paths
    yolo_path = r"runs\detect\GPU_Results\training_run_FP32-2\weights\best.pt"
    wpod_path = r"weights\wpodnet.pth"  # Using your local PyTorch weights!

    # 2. Spin up Thread 1 (Capture)
    cap_thread = threading.Thread(target=capture_frames, args=(rtsp_url,), daemon=True)
    cap_thread.start()

    # 3. Spin up Thread 2 (Detection) - Pass the 2 PyTorch paths
    det_thread = threading.Thread(
        target=process_detection, 
        args=(yolo_path, wpod_path), 
        daemon=True
    )
    det_thread.start()

    print("[MAIN] Opening display window...")

    # 4. Main loop handles ONLY the UI
    while running:
        if not display_queue.empty():
            final_frame = display_queue.get()
            cv2.imshow("Live RTSP Pipeline", final_frame)
        
        # UI refresh and quit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[MAIN] Shutting down...")
            running = False
            break

    # Clean shutdown
    cap_thread.join()
    det_thread.join()
    cv2.destroyAllWindows()
    print("[MAIN] System offline.")

if __name__ == "__main__":
    main()