# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 20:00:20 2025

@author: Parrain
"""
import cv2 as cv
import sys

fp = "D:/Intelligence Artificielle/Fichiers source/02/haarcascade_frontalface_default.xml"
face_cascade = cv.CascadeClassifier(fp)
fpIm = "D:/Intelligence Artificielle/Fichiers source/02/brad-angelina.jpg"

img = cv.imread(fpIm)
if img is None:
    sys.exit("Image non trouvée ou chemin incorrect !")

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(img_gray, 1.1, 8)

if len(faces) != 2:
    sys.exit("La photo doit avoir exactement 2 visages !")

x1, y1, w1, h1 = faces[0]
x2, y2, w2, h2 = faces[1]

face1 = img[y1:y1+h1, x1:x1+w1]
face2 = img[y2:y2+h2, x2:x2+w2]

#redimensionner 
face2 = cv.resize(face2, (w1,h1))
face1 = cv.resize(face1, (w2,h2))

img[y2:y2+h2, x2:x2+w2] = face1
img[y1:y1+h1, x1:x1+w1] = face2

cv.imshow('echange', img)
cv.waitKey(0)
cv.destroyAllWindows()