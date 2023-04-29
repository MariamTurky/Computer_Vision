import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
#################################################################################################################

def rgb_to_gray(source: np.ndarray):
    gray = np.dot(source[..., :3], [0.299, 0.587, 0.114]).astype('uint8')
    return gray
#################################################################################################################


def global_threshold(source: np.ndarray, threshold: int):
    # source: gray image
    src = np.copy(source)
    for x in range(src.shape[0]):
        for y in range(src.shape[1]):
            if src[x, y] > threshold:
                src[x, y] = 255
            else:
                src[x, y] = 0
    return src
    # src = np.copy(source)
    # if len(src.shape)> 2:
    #     src = rgb_to_gray(source)
    # return (src > threshold).astype('int')
#################################################################################################################


def local_threshold1(source: np.ndarray, divs: int):
    # source: gray image
    src = np.copy(source)
    for row in range(0, src.shape[0], divs):
        for col in range(0, src.shape[1], divs):
            mask_src = src[row:row+divs, col:col+divs]
            threshold = int(np.mean(mask_src))-10
            src[row:row+divs, col:col +
                divs] = global_threshold(source=mask_src, threshold=threshold)
    return src

#################################################################################################################


def histogram(source: np.array, bins_num: int = 255):
    if bins_num == 2:
        new_data = source
    else:
        new_data = source.astype('uint8')
    bins = np.arange(0, bins_num)
    hist = np.bincount(new_data.ravel(), minlength=bins_num)
    return hist, bins

#################################################################################################################


def equalize_histogram(source: np.ndarray, bins_num: int = 256):
    #
    bins = np.arange(0, bins_num)

    # Calculate the Occurrences of each pixel in the input
    hist_array = np.bincount(source.flatten(), minlength=bins_num)

    # Normalize Resulted array
    px_count = np.sum(hist_array)
    hist_array = hist_array / px_count

    # Calculate the Cumulative Sum
    hist_array = np.cumsum(hist_array)

    # Pixel Mapping
    trans_map = np.floor(255 * hist_array).astype('uint8')

    # Transform Mapping to Image
    img1d = list(source.flatten())
    map_img1d = [trans_map[px] for px in img1d]

    # Map Image to 2d & Reshape Image
    img_eq = np.reshape(np.asarray(map_img1d), source.shape)

    return img_eq, bins

#################################################################################################################


def normalize_histogram(source: np.ndarray, bins_num: int = 256):
    mn = np.min(source)
    mx = np.max(source)
    norm = ((source - mn) * (256 / (mx - mn))).astype('uint8')
    hist, bins = histogram(norm, bins_num=bins_num)
    return norm, hist, bins

#################################################################################################################


def draw_rgb_histogram(source: np.ndarray):

    colors = ["red", "green", "blue"]
    # colors =  [(0, 0, 1),(0, 1, 0),(1, 0, 0)]
    figure, axis = plt.subplots()
    for i in range(source.shape[2]):
        hist, bins = histogram(source=source[:, :, i], bins_num=256)
        # plt.plot(bins, hist, color=colors[i])
        axis.plot(bins, hist, color=colors[i])
        axis.set_xlabel('color value')
        axis.set_ylabel('Pixel count')
    st.pyplot(figure)

#################################################################################################################


def draw_gray_histogram(source: np.ndarray, bins_num):

    figure, axis = plt.subplots()
    # Create histogram and plot it
    hist, bins = histogram(source=source, bins_num=bins_num)
    axis.plot(bins, hist)
    axis.set_xlabel('gray value')
    axis.set_ylabel('Pixel count')
    st.pyplot(figure)

#################################################################################################################


def display_bar_graph(x, height, width):
    figure, axis = plt.subplots()
    plt.bar(x, height, width)
    st.pyplot(figure)
#################################################################################################################
def hist_bar(source: np.ndarray):
    figure, axis = plt.subplots()
    _ = plt.hist(source[:, :, 0].ravel(), bins = 256, color = 'red')
    _ = plt.hist(source[:, :, 1].ravel(), bins = 256, color = 'Green')
    _ = plt.hist(source[:, :, 2].ravel(), bins = 256, color = 'Blue')
    _ = plt.xlabel('Intensity Value')
    _ = plt.ylabel('Count')
    st.pyplot(figure)
#################################################################################################################

def rgb_distribution_curve(source: np.ndarray):
    colors = ["red", "green", "blue"]
    figure, axis = plt.subplots()
    for i in range(source.shape[2]):
        hist, bins = histogram(source=source[:, :, i], bins_num=256)
        pdf = (hist) / sum(hist)
        cdf = np.cumsum(pdf)
        axis.plot(bins, cdf, label="CDF", color=colors[i])
    st.pyplot(figure)
#############################################################################################################