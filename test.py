import os
import cv2
import numpy as np
from ultralytics import YOLO  # Ensure this matches YOLOv11's API

# Set environment variables to avoid OpenMP errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Allow multiple OpenMP runtimes
os.environ["OMP_NUM_THREADS"] = "1"  # Restrict OpenMP threads

def load_model(weights_path):
    """
    Load YOLO model with specified weights.
    """
    try:
        model = YOLO(weights_path)  # Load YOLO model
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def process_frame(model, frame):
    """
    Perform inference on a frame using the YOLO model.
    """
    results = model(frame)  # Inference
    detections = results[0].boxes.data.cpu().numpy()  # Extract detection data
    return detections, results[0].names  # Return detections and class names

def draw_detections(frame, detections, class_names):
    """
    Draw detections on the frame with bounding boxes and labels.
    """
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        label = f"{class_names[int(cls)]} {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Add label
        cv2.putText(
            frame,
            label,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    return frame

def main(weights_path):
    """
    Main function to process webcam feed and display real-time predictions.
    """
    # Load YOLO model
    model = load_model(weights_path)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform inference
        detections, class_names = process_frame(model, frame)

        # Draw detections on the frame
        frame = draw_detections(frame, detections, class_names)

        # Display the frame
        cv2.imshow('YOLOv11 Real-Time Inference', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    weights_path = "best.pt"  # Replace with your YOLOv11 weights file
    main(weights_path)