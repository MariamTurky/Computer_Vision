import numpy as np
import cv2
import matplotlib.pyplot as plt
#############################################################################################################
def harrisCorner(image ,k):
    src = np.copy(image)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    Ix = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=5)
    Ixx = cv2.GaussianBlur(src=Ix ** 2, ksize=(5, 5), sigmaX=0)
    Ixy = cv2.GaussianBlur(src=Iy * Ix, ksize=(5, 5), sigmaX=0)
    Iyy = cv2.GaussianBlur(src=Iy ** 2, ksize=(5, 5), sigmaX=0)

    det = Ixx *Iyy - (Ixy**2)
    trace = Ixx + Iyy
    harrisResponse = det - k * (trace **2)

    return harrisResponse
#############################################################################################################
def corner2Image (image, harrisResponse , cornerThreshold = 0.01 ):
    cop_harris = np.copy(harrisResponse)
    harrisMatrix =cv2.dilate(cop_harris,None)
    hMax  = harrisMatrix.max()
    corner_indices = np.array(harrisMatrix > (hMax * cornerThreshold), dtype="int8")
    # corner_indices = corner_indices[2:,2:]
    image[corner_indices == 1 ] = [0 , 255 ,0]
    return image    
#############################################################################################################