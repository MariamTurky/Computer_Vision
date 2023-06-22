import numpy as np
import cv2
import matplotlib.pyplot as plt
#################################################################################################################
def BGR_To_LUV(image):
    copied_image = image.copy()
    width, height = copied_image.shape[:2]
    copied_image =copied_image /255.0

    for i in range(width):
        
        for j in range(height):
            
            x = 0.412453 * copied_image[i, j][0] + 0.357580 * copied_image[i, j][1] + 0.180423 * copied_image[i, j][2]
            y = 0.212671 * copied_image[i, j][0] + 0.715160 * copied_image[i, j][1] + 0.072169 * copied_image[i, j][2]
            z = 0.019334 * copied_image[i, j][0] + 0.119193 * copied_image[i, j][1] + 0.950227 * copied_image[i, j][2]

            if (y > 0.008856):
                L = (116.0 * (y **(1/3)) ) - 16.0
            else:
                L = 903.3 * y
            
            u_dash = 4.0*x /( x + (15.0*y ) + 3.0*z) 
            v_dash = 9.0*y /( x + (15.0*y ) + 3.0*z) 

            U = 13.0 * L * (u_dash -0.19793943)
            V = 13.0 * L * (v_dash -0.46831096)

            image [i,j] [0] = ( 255.0/100.0) *L
            image [i,j] [1] = ( 255.0/ 354.0) *(U+134.0)
            image [i,j] [2] = (255.0/ 262.0) *(V +140.0) 

    return image.astype(np.uint8)
#################################################################################################################