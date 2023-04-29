import numpy as np
from math import sqrt, atan2, pi
import matplotlib.pyplot as plt
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
#############################################################################################################
def hough_ellipse(img, threshold=4, accuracy=1, min_size=4, max_size=None):
    image_rgb = img
    img = color.rgb2gray(img)
    img = canny(img, sigma=2.0,
              low_threshold=0.55, high_threshold=0.8)

    
    if img.ndim != 2:
        raise ValueError('The input image must be 2D.')

    if not np.any(img):
        return np.zeros((0, 6))

    pixels = np.row_stack(np.nonzero(img))

    num_pixels = pixels.shape[1]
    acc = []
    results = []
    bin_size = accuracy * accuracy

    if max_size is None:
        if img.shape[0] < img.shape[1]:
            max_b_squared = np.round(0.5 * img.shape[0])
        else:
            max_b_squared = np.round(0.5 * img.shape[1])
        max_b_squared *= max_b_squared
    else:
        max_b_squared = max_size * max_size

    for p1 in range(num_pixels):
        p1x = pixels[1, p1]
        p1y = pixels[0, p1]

        for p2 in range(p1):
            p2x = pixels[1, p2]
            p2y = pixels[0, p2]

            dx = p1x - p2x
            dy = p1y - p2y
            a = 0.5 * sqrt(dx * dx + dy * dy)
            if a > 0.5 * min_size:
                xc = 0.5 * (p1x + p2x)
                yc = 0.5 * (p1y + p2y)

                for p3 in range(num_pixels):
                    p3x = pixels[1, p3]
                    p3y = pixels[0, p3]
                    dx = p3x - xc
                    dy = p3y - yc
                    d = sqrt(dx * dx + dy * dy)
                    if d > min_size:
                        dx = p3x - p1x
                        dy = p3y - p1y
                        cos_tau_squared = ((a*a + d*d - dx*dx - dy*dy) / (2 * a * d))
                        cos_tau_squared *= cos_tau_squared

                        k = a*a - d*d * cos_tau_squared
                        if k > 0 and cos_tau_squared < 1:
                            b_squared = a*a * d*d * (1 - cos_tau_squared) / k

                            if b_squared <= max_b_squared:
                                acc.append(b_squared)

                if len(acc) > 0:
                    bins = np.arange(0, np.max(acc) + bin_size, bin_size)
                    hist, bin_edges = np.histogram(acc, bins=bins)
                    hist_max = np.max(hist)
                    if hist_max > threshold:
                        orientation = atan2(p1x - p2x, p1y - p2y)
                        b = sqrt(bin_edges[hist.argmax()])
                        if orientation != 0:
                            orientation = pi - orientation
                            if orientation > pi:
                                orientation = orientation - pi / 2.
                                a, b = b, a
                        results.append((hist_max, yc, xc, a, b, orientation))
                    acc = []

    result = np.array(results, dtype=[('accumulator', np.intp),
                                    ('yc', np.float64),
                                    ('xc', np.float64),
                                    ('a', np.float64),
                                    ('b', np.float64),
                                    ('orientation', np.float64)])

    result.sort(order='accumulator')
    best = list(result[-1])
    yc, xc, a, b = (int(round(x)) for x in best[1:5])
    orientation = best[5]



    # Draw the ellipse on the original image with thicker lines
    for offset in range(-2, 3):
        cy, cx = ellipse_perimeter(yc + offset, xc + offset, a, b, orientation)
        image_rgb[cy, cx] = (0, 0, 255)


    return image_rgb
#############################################################################################################   

# image_rgb = data.coffee()[0:220, 160:420]
# image_gray = color.rgb2gray(image_rgb)
# edges = canny(image_gray, sigma=2.0,
#               low_threshold=0.55, high_threshold=0.8)
# result = hough_ellipse(edges, accuracy=20, threshold=250,
#                        min_size=100, max_size=120)
# result.sort(order='accumulator')



# ### super impose


# # Estimated parameters for the ellipse
# best = list(result[-1])
# yc, xc, a, b = (int(round(x)) for x in best[1:5])
# orientation = best[5]



# # Draw the ellipse on the original image with thicker lines
# for offset in range(-2, 3):
#     cy, cx = ellipse_perimeter(yc + offset, xc + offset, a, b, orientation)
#     image_rgb[cy, cx] = (0, 0, 255)

# # Display the image with the thicker ellipse
# plt.imshow(image_rgb)
# plt.show()
#############################################################################################################