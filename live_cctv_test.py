import cv2
import threading
import queue
import time
import os
from ultralytics import YOLO

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
 


def process_detection(model_path):
    """THREAD 2: YOLO AI (Consumer & Producer)"""
    global running
    print("[THREAD 2] Loading YOLO model...")
    
    # Load model inside the thread that will use it
    model = YOLO(model_path)
    
    while running:
        if not frame_queue.empty():
            # 1. Pull raw frame
            frame = frame_queue.get()

            # 2. Run Heavy AI Math
            results = model.predict(source=frame, conf=0.5, verbose=False)
            
            # 3. Draw bounding boxes
            annotated_frame = results[0].plot()

            # 4. Push to UI queue
            if display_queue.full():
                display_queue.get()
            display_queue.put(annotated_frame)
        else:
            # Brief pause to prevent CPU maxing out if queue is empty
            time.sleep(0.01)

    print("[THREAD 2] Detection stopped.")

def main():
    """MAIN THREAD: UI Display (Consumer)"""
    global running
    print("--- Starting Multi-Threaded Pipeline ---")

    #camera_source=0
    rtsp_url = 'rtsp://admin:It%40123aasl@10.64.64.16/Streaming/channels/001/?transportmode=unicast'
    model_path = r"runs\detect\GPU_Results\training_run_FP32-2\weights\best.pt"

    # Spin up Thread 1 (Capture)
    cap_thread = threading.Thread(target=capture_frames, args=(rtsp_url,), daemon=True)
    cap_thread.start()

    # Spin up Thread 2 (Detection)
    det_thread = threading.Thread(target=process_detection, args=(model_path,), daemon=True)
    det_thread.start()

    print("[MAIN] Opening display window...")

    # Main loop handles ONLY the UI
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