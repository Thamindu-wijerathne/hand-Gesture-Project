import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20  # Adjust the offset as needed
imgSize = 300

folder = "D:\Testing\Pycharm\handGesture\handGestureProject\Data\D"
counter = 0

# Ensure the folder exists before saving
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    if not success:
        break

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

        #create black page to put skeloton into that
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        # Create a bounding box and display cropped image
        imgCrop = img[y1:y2, x1:x2]

        #put cropped img onto white page
        imgCropShape = imgCrop.shape

        aspectRatio = h/w #if more than 1 height is greater

        if aspectRatio >1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
        else :
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (hCal, imgSize))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[:, hGap:hCal+hGap] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Display the original image
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        file_path = f'{folder}/Image_{time.time()}.jpg'
        try:
            # Attempt to save the image
            success = cv2.imwrite(file_path, imgWhite)
            if success:
                print(f"Image saved successfully: {file_path}")
            else:
                print("Error: Image could not be saved.")
        except Exception as e:
            print(f"Exception occurred while saving image: {e}")

    # 1 millisecond delay; press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
