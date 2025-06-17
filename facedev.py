# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 08:29:58 2025

@author: Parrain
"""

import cv2 as cv
fp = "D:/Intelligence Artificielle/Fichiers source/02/haarcascade_frontalface_default.xml"
fpIm = "D:/Intelligence Artificielle/Fichiers source/02/obama.jpg"
fpEy = "D:/Intelligence Artificielle/Fichiers source/02/haarcascade_eye.xml"
face_cascade = cv.CascadeClassifier(fp)
eye_cascade = cv.CascadeClassifier(fpEy)

img = cv.imread(fpIm)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 8)


for face in faces:
    x, y, w, h = face 
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
for (ex, ey, ew, eh) in eyes:
    cv.rectangle(img, (ex, ey),(ex + ew, ey + eh), (255,0,0), 2)
    
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()