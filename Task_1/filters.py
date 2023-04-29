import cv2
import numpy as np
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from PIL import Image
from scipy.signal import convolve2d
#############################################################################################################
def add_gaussian_noise(image, mean=0, var=0.01):
    image = image/255
    sigma = np.sqrt(var)
    noise = np.random.normal(mean, sigma, size=image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image
#############################################################################################################
# blank image


def add_salt_pepper_noise(image, pepper_amount=0):
    image = image/255

    salt_amount = 1 - pepper_amount
    noisy_image = np.copy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < pepper_amount:
                noisy_image[i][j] = 0
            elif rdn > salt_amount:
                noisy_image[i][j] = 1
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image
#############################################################################################################
# uniform noise


def add_uniform_noise(image, a=0, b=0.2):
    image = image/255

    noise = np.random.uniform(a, b, size=image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image
#############################################################################################################

def gaussian_filter(img, D0=20):
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    x, y = img.shape
    H = np.zeros((x, y), dtype=np.float32)
    for i in range(x):
        for j in range(y):
            D = np.sqrt((i-x/2)**2 + (j-y/2)**2)
            H[i, j] = np.exp(-D**2/(2*D0*D0))
    Gshift = Fshift * H
    G = np.fft.ifftshift(Gshift)
    g_filter = np.abs(np.fft.ifft2(G))
    return g_filter
#############################################################################################################


def average_filter(img):
    mask = np.ones([3, 3], dtype=int)
    mask = mask / 9
    x, y = img.shape
    img_new = np.zeros([x, y])
    for i in range(1, x-1):
        for j in range(1, y-1):
            temp = img[i-1, j-1]*mask[0, 0] + img[i-1, j]*mask[0, 1] + img[i-1, j + 1]*mask[0, 2] + img[i, j-1]*mask[1, 0] + img[i, j] * \
                mask[1, 1] + img[i, j + 1]*mask[1, 2] + img[i + 1, j-1]*mask[2,
                                                                             0] + img[i + 1, j]*mask[2, 1] + img[i + 1, j + 1]*mask[2, 2]
            img_new[i, j] = temp
    img_new = img_new.astype(np.uint8)
    return img_new
#############################################################################################################

def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data), len(data[0])))
    for i in range(len(data)):
        for j in range(len(data[0])):
            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])
            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final
#############################################################################################################

def convolution(image, kernel):
    """
    Perform 2D convolution of an image with a given kernel by flipping the kernel and then using the scipy.signal.convolve2d function.
    Clip the resulting image values to 0-255 and convert to unsigned 8-bit integers.

    Parameters:
        image (numpy.ndarray): 2D grayscale image array
        kernel (numpy.ndarray): 2D kernel array

    Returns:
        numpy.ndarray: 2D array of unsigned 8-bit integers representing the convolved image
    """
    conv_im = convolve2d(image, kernel[::-1, ::-1]).clip(0, 255)
    conv_im = conv_im.astype(np.uint8)

    return conv_im
#############################################################################################################

def zero_padding(image):
    """
    Add zero padding of 1 pixel to the top, bottom, left, and right of the image by creating a new array of zeros with dimensions 2 pixels larger than the image,
    and copying the original image into the center of the new array.

    Parameters:
        image (numpy.ndarray): 2D grayscale image array

    Returns:
        numpy.ndarray: 2D array of unsigned 8-bit integers representing the padded image
    """
    image_dimensions = image.shape
    padded_image = np.zeros(
        (image_dimensions[0]+2, image_dimensions[1] + 2), dtype=np.uint8)

    for i in range(image_dimensions[0]):
        for j in range(image_dimensions[1]):
            padded_image[i+1, j+1] = image[i, j]

    return padded_image
#############################################################################################################

def non_maximum_suppression(gradient_direction, gradient_magnitude, image, nms_thresholding=0.1):
    image = np.zeros_like(gradient_magnitude)
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            angle = gradient_direction[i, j]
            if (angle < 0):
                angle += 180
            if (angle < 22.5 or angle > 157.5):
                if (gradient_magnitude[i, j] > gradient_magnitude[i, j-1] and gradient_magnitude[i, j] > gradient_magnitude[i, j+1]):
                    image[i, j] = gradient_magnitude[i, j]
            elif (22.5 <= angle < 67.5):
                if (gradient_magnitude[i, j] > gradient_magnitude[i-1, j-1] and gradient_magnitude[i, j] > gradient_magnitude[i+1, j+1]):
                    image[i, j] = gradient_magnitude[i, j]
            elif (67.5 <= angle < 112.5):
                if (gradient_magnitude[i, j] > gradient_magnitude[i-1, j] and gradient_magnitude[i, j] > gradient_magnitude[i+1, j]):
                    image[i, j] = gradient_magnitude[i, j]
            elif (112.5 <= angle < 157.5):
                if (gradient_magnitude[i, j] > gradient_magnitude[i-1, j+1] and gradient_magnitude[i, j] > gradient_magnitude[i+1, j-1]):
                    image[i, j] = gradient_magnitude[i, j]

    # perform thresholding
    image = image / np.max(image)

    image[image < nms_thresholding] = 0
    image[image >= nms_thresholding] = 1
    image = image * 255
    return image
#############################################################################################################

def double_threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):

    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)
#############################################################################################################

def hysteresis(img, weak, strong):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i, j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img
#############################################################################################################

def edge_detection(image, detector='canny'):
    """
    Apply edge detection on the image located at `image_path` using the specified detector.

    Parameters:
    -----------
    image_path: str
        The path to the image file to be processed.
    detector: str, optional
        The type of edge detector to be used. Supported values are 'canny' (default), 'sobel', 'roberts', and 'prewitt'.

    Returns:
    --------
    Image
    """
    # image = cv2.imread(image_path, 0)
    padded_image = zero_padding(image)
    mean_kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
    x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    smoothed_image = convolution(padded_image, mean_kernel)
    returned_image = 0

    if detector == 'canny':
        x_edges = convolution(smoothed_image, x_kernel)
        y_edges = convolution(smoothed_image, y_kernel)
        gradient_magnitude = np.hypot(x_edges, y_edges)
        gradient_direction = np.degrees(np.arctan2(y_edges, x_edges))
        gradient_magnitude = gradient_magnitude.astype(np.float32)
        edges_image = non_maximum_suppression(
            gradient_direction, gradient_magnitude, image, 0.2)
        thresholded, weak, strong = double_threshold(edges_image)
        final_image = hysteresis(thresholded, weak, strong)
        returned_image = final_image

    elif detector == 'sobel':
        x_edges = convolution(smoothed_image, x_kernel)
        y_edges = convolution(smoothed_image, y_kernel)
        edges_image_sobel = np.hypot(x_edges, y_edges)
        edges_image_sobel = edges_image_sobel.astype(np.uint8)
        returned_image = edges_image_sobel

    elif detector == 'roberts':
        kernel_x = np.array([[1, 0], [0, -1]])
        kernel_y = np.array([[0, 1], [-1, 0]])
        horizontal_detection = convolution(smoothed_image, kernel_x)
        vertical_detection = convolution(smoothed_image, kernel_y)
        edges_image_roberts = np.hypot(
            horizontal_detection, vertical_detection)
        edges_image_roberts = edges_image_roberts.astype(np.uint8)
        returned_image = edges_image_roberts

    elif detector == 'prewitt':
        kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        horizontal_detection = convolution(smoothed_image, kernel_x)
        vertical_detection = convolution(smoothed_image, kernel_y)
        edges_image_prewitt = np.hypot(
            horizontal_detection, vertical_detection)
        edges_image_prewitt = edges_image_prewitt.astype(np.uint8)
        returned_image = edges_image_prewitt

    else:
        raise ValueError("Invalid detector specified")

    return returned_image

#############################################################################################################

