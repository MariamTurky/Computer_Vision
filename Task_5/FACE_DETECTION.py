import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import face

def convertToRGB(image):
    
    if image is None:
        raise ValueError("Input image is empty")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def detect_faces(image):

    # Load the pre-trained classifiers for face
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces=face_cascade.detectMultiScale(image, scaleFactor=1.2,minNeighbors=5)

    return faces


def draw_faces(image, faces):

     # Draw a rectangle around the faces
    for (x, y, w, h) in faces: 
        cv2.rectangle(image, (x, y), (x+w, y+h),(0, 255, 0), 1)


    return image

