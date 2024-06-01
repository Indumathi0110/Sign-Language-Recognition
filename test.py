import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier(r"C:\Users\HOPEWORKS\Desktop\models\keras_model.h5", r"C:\Users\HOPEWORKS\Desktop\models\labels.txt")
offset = 20
counter = 0

labels = ["friend","hello","help","iloveyou","no","please","sorry","stop","thankyou","yes"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = img.shape[0] / imgCropShape[0]  # Calculating the aspect ratio based on height
            wCal = math.ceil(k * imgCropShape[1])  # Calculating the width
            imgResize = cv2.resize(imgCrop, (wCal, img.shape[0]))  # Resizing the image
            prediction , index = classifier.getPrediction(imgResize, draw=False)  # Using resized image for prediction
            print(prediction, index)
        else:
            k = img.shape[1] / imgCropShape[1]  # Calculating the aspect ratio based on width
            hCal = math.ceil(k * imgCropShape[0])  # Calculating the height
            imgResize = cv2.resize(imgCrop, (img.shape[1], hCal))  # Resizing the image
            prediction , index = classifier.getPrediction(imgResize, draw=False)  # Using resized image for prediction
            print(prediction, index)

        cv2.rectangle(imgOutput,(x-offset,y-offset-70),(x -offset+400, y - offset+60-50),(0,255,0),cv2.FILLED)  
        cv2.putText(imgOutput,labels[index],(x,y-30),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2) 
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x + w + offset, y+h + offset),(0,255,0),4)   

        # Comment out or remove the following lines to remove unnecessary image tabs
        # cv2.imshow('ImageCrop', imgCrop)
        # cv2.imshow('ImageResize', imgResize)

    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)
