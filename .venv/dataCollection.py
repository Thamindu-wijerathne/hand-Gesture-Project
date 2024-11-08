import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import subprocess
import time
import os  # for process termination

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
Classifier = Classifier("D:/Testing/Pycharm/handGesture/handGestureProject/Model/keras_model.h5",
                        "D:/Testing/Pycharm/handGesture/handGestureProject/Model/labels.txt")

offset = 20  # Adjust the offset as needed
imgSize = 300

labels = ["A", "B", "C", "D"]

# Define a dictionary to map gestures to app paths
app_dict = {
    "A": "C:/Users/Thamindu/Desktop/1.png",
    "B": "C:/Users/Thamindu/Desktop/2.png",
    "C": "C:/Users/Thamindu/Desktop/3.png",
}

# Variables to track gesture time and the last opened app
gesture_time = None
current_gesture = None
last_opened_process = None

while True:
    success, img = cap.read()
    if not success:
        break

    index = -1  # Initialize index to a default value
    imgOutput = img.copy()

    # Detect hands in the image
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Apply offset and ensure coordinates are within the image bounds
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        # Create white background to hold the skeleton
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand from the frame
        imgCrop = img[y1:y2, x1:x2]
        aspectRatio = h / w  # If aspect ratio is more than 1, height is greater

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = Classifier.getPrediction(imgWhite, draw=False)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (hCal, imgSize))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[:, hGap:hCal + hGap] = imgResize
            prediction, index = Classifier.getPrediction(imgWhite, draw=False)

        # Inside the while loop, update this section:
        if 0 <= index < len(labels):
            new_gesture = labels[index]

            # Only process if the detected gesture has changed from the last gesture
            if new_gesture != current_gesture:
                current_gesture = new_gesture  # Set the current gesture to the new one
                gesture_time = time.time()  # Reset the timer for the new gesture

            # Check if the gesture is stable for 3 seconds before launching
            if new_gesture == current_gesture and time.time() - gesture_time >= 3:
                # Check if the gesture corresponds to an app in app_dict and last_opened_process is None
                if new_gesture in app_dict and (not last_opened_process or last_opened_process.poll() is not None):
                    app_path = app_dict[new_gesture]
                    last_opened_process = subprocess.Popen([app_path], shell=True)
                    print(f"Opening {new_gesture} App...")

                    # After opening, reset gesture tracking so it doesn’t open repeatedly
                    current_gesture = None
                    gesture_time = None  # Reset gesture time to prevent re-opening

            # If "D" gesture is detected, close the last opened app
            if new_gesture == "D" and last_opened_process and last_opened_process.poll() is None:
                try:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(last_opened_process.pid)], shell=True)
                    print("Closed the last opened app.")
                    last_opened_process = None  # Reset after closing the app
                except Exception as e:
                    print(f"Error closing app: {e}")

            # Display the recognized label on the screen
            cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

    # Display the original image
    cv2.imshow("Image", imgOutput)

    # 1 millisecond delay; press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
