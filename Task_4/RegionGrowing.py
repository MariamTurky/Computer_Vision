import numpy as np
import cv2

def region_growing(image, seed):
    # Create a mask with the same shape as the input image
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    # Define the seed point and set its value as the initial threshold
    x, y = seed
    threshold = image[x, y]
    
    # Create a queue to hold the pixels to be processed
    queue = []
    queue.append((x, y))

    # Loop through the queue until it is empty
    while queue:
        # Get the next pixel from the queue
        x, y = queue.pop(0)

        # Check if the pixel is within the image bounds
        if x < 0 or y < 0 or x >= h or y >= w:
            continue

        # Check if the pixel has already been processed
        if mask[x, y]:
            continue

        # Check if the pixel's intensity is within the threshold
        if abs(image[x, y] - threshold).any() <= 10:
            # Add the pixel to the mask and queue
            mask[x, y] = 255
            queue.append((x - 1, y))
            queue.append((x + 1, y))
            queue.append((x, y - 1))
            queue.append((x, y + 1))
    
    return mask