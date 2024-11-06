import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import os
import subprocess

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
Classifier = Classifier("D:\Testing\Pycharm\handGesture\handGestureProject\Model\keras_model.h5", "D:\Testing\Pycharm\handGesture\handGestureProject\Model\labels.txt")


offset = 20  # Adjust the offset as needed
imgSize = 300

labels = ["A", "B", "C", "D"]

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
            prediction, index = Classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else :
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (hCal, imgSize))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[:, hGap:hCal+hGap] = imgResize
            prediction, index = Classifier.getPrediction(imgWhite, draw=False)


        # Display the label only if `index` is valid (cuz if hand is regocnize and if it not in Prediction then cause error)
        if 0 <= index < len(labels):
            #cv2.rectangle(imgOutput, (x-offset,y-offset-50), (x-offset+150, y-offset-50+50), (255,0,255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
            #cv2.rectangle(imgOutput, (x-offset,y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)

        #cv2.imshow("ImageCrop", imgCrop)
        #cv2.imshow("ImageWhite", imgWhite)

    # Display the original image
    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)

    # 1 millisecond delay; press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
