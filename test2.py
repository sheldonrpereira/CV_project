from ultralytics import YOLO
import cv2

def detect_objects(model_path='C:\\Users\\sheld\\Desktop\\uni\\cv_project\\runs\\runs\\detect\\emotion_detection_training\\weights\\best.pt', conf_thresh=0.35, device=0):
    """
    Real-time detection of objects using a custom YOLO model with class labels.
    """
    try:
        model = YOLO(model_path)
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            raise ValueError("Could not open webcam")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("\nStarting detection... Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            results = model(frame, conf=conf_thresh, verbose=False)
            annotated_frame = frame.copy()

            if len(results) > 0:  # Check if there are any results
                boxes = results[0].boxes  # Get boxes from first result
                for box in boxes:
                    cls_id = int(box.cls[0])  # Get class ID
                    class_name = model.names[cls_id]  # Get class name from model

                    # Get coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    color = (0, 255, 0)  # Green for detected objects

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    conf = float(box.conf[0])
                    label_text = f"{class_name} {conf:.2f}"
                    cv2.putText(annotated_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow('Object Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")

if __name__ == "__main__":
    detect_objects()