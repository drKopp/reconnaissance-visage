# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 08:29:58 2025

@author: Parrain
"""

import cv2 as cv
fp = "D:/Tuto.com - Intelligence Artificielle - 5 projets complets et pratiques en Python/Fichiers source/02/haarcascade_frontalface_default.xml"
fpIm = "D:/Tuto.com - Intelligence Artificielle - 5 projets complets et pratiques en Python/Fichiers source/02/obama.jpg"
face_cascade = cv.CascadeClassifier(fp)

img = cv.imread(fpIm)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 8)

i = 0
for face in faces:
    
    
    x, y, w, h = face 
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    face = img[y:y+ h, x:x+w]
    cv.imshow('face{}'.format(i), face)
    i += 1
    
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()