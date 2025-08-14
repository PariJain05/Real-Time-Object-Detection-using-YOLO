import cv2
import pyttsx3
from ultralytics import YOLO
import tkinter as tk
from tkinter import Button, Label
from threading import Thread
import time

# -------------------- Initialize text-to-speech engine --------------------
engine = pyttsx3.init()

# -------------------- Load YOLOv8 model --------------------
# You can replace "yolov8n.pt" with your own trained model file
model = YOLO("yolov8n.pt")

# -------------------- Global variables --------------------
cap = None
running = False


# -------------------- Speak detected object --------------------
def speak(text):
    """Speak out the detected label."""
    engine.say(text)
    engine.runAndWait()


# -------------------- Object detection function --------------------
def detect_objects():
    """Detect objects using YOLO and display results."""
    global running, cap

    # Open the camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame for faster processing
        resized_frame = cv2.resize(frame, (640, 480))

        # Perform detection
        results = model.predict(resized_frame, conf=0.5, device="cpu", verbose=False)

        # Parse the results
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = box.conf.item()
                label = model.names[class_id]

                # Skip "person" class if not needed
                if label.lower() == "person":
                    continue

                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 255, 0)

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Speak the detection
                speak(f"Detected {label} at position {x1}, {y1}")

        # Display the frame
        cv2.imshow("YOLOv8 Detection", frame)

        # Add a delay to reduce CPU usage
        time.sleep(0.1)

        # Exit loop if 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------- Start/Stop functions --------------------
def start_detection():
    global running
    if not running:
        running = True
        Thread(target=detect_objects).start()


def stop_detection():
    global running, cap
    running = False
    if cap:
        cap.release()
    cv2.destroyAllWindows()


# -------------------- Create GUI --------------------
root = tk.Tk()
root.title("YOLO Object Detection")
root.geometry("400x200")

Label(root, text="YOLO Object Detection", font=("Arial", 16)).pack(pady=10)
Button(root, text="Start Detection", command=start_detection,
       font=("Arial", 12), bg="green", fg="white").pack(pady=5)
Button(root, text="Stop Detection", command=stop_detection,
       font=("Arial", 12), bg="red", fg="white").pack(pady=5)
Label(root, text="Press 'Esc' to stop detection in the video feed.",
      font=("Arial", 10)).pack(pady=10)

# -------------------- Run the GUI --------------------
root.mainloop()
