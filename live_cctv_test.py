import cv2
from ultralytics import YOLO

def main():
    print("--- Starting Live CCTV Feed ---")

    # 1. Load your highly accurate trained model
    model_path = r"runs\detect\GPU_Results\training_run_FP32-2\weights\best.pt"
    model = YOLO(model_path)
    print("Model loaded successfully!")

    # 2. Define the camera URL
    camUrl = 'rtsp://admin:It%40123aasl@10.64.64.16/Streaming/channels/001/?transportmode=unicast'

    # 3. Start a continuous loop to fetch frames one by one
    while True:
        try:
            # Connect to the camera and grab a single picture
            vs = cv2.VideoCapture(camUrl)
            ret, frame = vs.read()
            vs.release() # Close the connection immediately after grabbing the frame

            # If the frame was successfully grabbed, process it
            if ret and frame is not None:
                
                # 4. Run YOLO inference on the grabbed frame
                # Setting conf=0.5 to keep false positives low in the real world
                results = model.predict(source=frame, conf=0.5, verbose=False)
                
                # 5. Draw the bounding boxes on the frame
                annotated_frame = results[0].plot()
                
                # 6. Display the live feed in a window
                cv2.imshow("Live CCTV - ANPR", annotated_frame)

            else:
                print("Failed to grab frame from camera.")

        except KeyError:
            print('Undefined input to listener')
        except Exception as e:
            print(f"Network or Camera Error: {e}")

        # 7. Add a way to cleanly exit the loop (Press 'q' to quit)
        # cv2.waitKey(1) pauses for 1 millisecond to let the window refresh
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Closing feed...")
            break

    # Clean up the window when the loop finishes
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()