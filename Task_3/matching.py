import SIFT as sift
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
#############################################################################################################
def matching(descriptor1 , descriptor2 , match_calculator):
    
    keypoints1 = descriptor1.shape[0]
    keypoints2 = descriptor2.shape[0]
    matches = []

    for kp1 in range(keypoints1):

        distance = -np.inf
        y_index = -1
        for kp2 in range(keypoints2):

         
            value = match_calculator(descriptor1[kp1], descriptor2[kp2])

            if value > distance:
              distance = value
              y_index = kp2
        
        match = cv2.DMatch()
        match.queryIdx = kp1
        match.trainIdx = y_index
        match.distance = distance
        matches.append(match)
    matches= sorted(matches, key=lambda x: x.distance, reverse=True)
    return matches
############################################################################################################# 

def calculate_ncc(descriptor1 , descriptor2):


    out1_normalized = (descriptor1 - np.mean(descriptor1)) / (np.std(descriptor1))
    out2_normalized = (descriptor2 - np.mean(descriptor2)) / (np.std(descriptor2))

    correlation_vector = np.multiply(out1_normalized, out2_normalized)

    correlation = float(np.mean(correlation_vector))

    return correlation
#############################################################################################################

def calculate_ssd(descriptor1 , descriptor2):

    ssd = 0
    for m in range(len(descriptor1)):
        ssd += (descriptor1[m] - descriptor2[m]) ** 2

    ssd = - (np.sqrt(ssd))
    return ssd

#############################################################################################################

def get_matching(img1_path,img2_path,method):
    img1 = cv2.imread(img1_path, 0)           # queryImage
    img2 = cv2.imread(img2_path, 0)  # trainImage

    # Compute SIFT keypoints and descriptors
    start_time =time.time()
    keypoints_1, descriptor1 = sift.SIFT(img1)

    keypoints_2, descriptor2 = sift.SIFT(img2)

    end_time = time.time()
    Duration_sift = end_time - start_time

    if method  == 'ncc':
        start = time.time()
        matches_ncc = matching(descriptor1, descriptor2, calculate_ncc)
        matched_image = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2,
                                        matches_ncc[:30], img2, flags=2)
        end = time.time()
        match_time = end - start

    else:

        start = time.time()

        matches_ssd = matching(descriptor1, descriptor2, calculate_ssd)
        matched_image = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2,
                                        matches_ssd[:30], img2, flags=2)
        end = time.time()
        match_time = end - start
    
    return matched_image , match_time
#############################################################################################################