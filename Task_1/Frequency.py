import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,exp
#################################################################################################################
##################################### Tab 3 (High and Low) ######################################################
#################################################################################################################
def prepare(path):
    image = cv2.imread(path)
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (210,210))

    return image
#################################################################################################################
#################################################################################################################
def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
def idealFilterHP(D0,imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 0
    return base
def idealFilterLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 1
    return base
def image_after_highpassfilter(path):
    img =prepare(path)
    original = np.fft.fft2(img)
    center = np.fft.fftshift(original) # Move frequency domain from upper left to middle
    
    #High pass filter
    HighPass = idealFilterHP(50,img.shape)
    HighPassCenter = center * idealFilterHP(15,img.shape)
    HighPass = np.fft.ifftshift(HighPassCenter)
    # Inverse Fourier Transform
    inverse_HighPass = np.fft.ifft2(HighPass)  # Fourier library function call
    ifimg1 = np.abs(inverse_HighPass)
    cv2.imwrite('img/image1.png',np.abs(ifimg1))

    return 'img/image1.png'

#################################################################################################################
def image_after_lowpassfilter(path):
    
    gray_image =prepare(path)
    original = np.fft.fft2(gray_image)
    center = np.fft.fftshift(original)

    # Low-pass filter
    LowPass = idealFilterLP(50,gray_image.shape)
    # Inverse Fourier Transform
    LowPassCenter = center * idealFilterLP(15,gray_image.shape)
    # rows,cols=gray_image.shape
    LowPass = np.fft.ifftshift(LowPassCenter)
    inverse_LowPass = np.fft.ifft2(LowPass)
    ifimg = np.abs(inverse_LowPass)
    cv2.imwrite('img/image2.png',np.abs(ifimg))
    return 'img/image2.png'

#################################################################################################################
def getfilter(path,flag):
    if flag == 1:
        return image_after_highpassfilter(path)
    else :
         return image_after_lowpassfilter(path)


def hybrid_images(path1 ,path2):

    image1 = prepare(path1)
    image2 = prepare(path2)
    new_img =  image1 + image2

    # Save the image 
    cv2.imwrite('img/hybrid_image.png',new_img)
    return 'img/hybrid_image.png'

#################################################################################################################