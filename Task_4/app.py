################################## Essential imports ######################################################
from turtle import color
import pandas as pd
import streamlit as st
from PIL import Image
import Histograms as Histogram
import numpy as np
import extra_streamlit_components as stx
import matplotlib.pyplot as plt
import plotly_express as px
from io import StringIO
import filters as fs
import os
import Frequency as freq
import Hough as Hough
import contour
import Thresholding as Thresholding
import luv as luv
import Kmeans_segmentation as km
import RegionGrowing as rg
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from cv2 import drawKeypoints, imread, imshow, resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST, waitKey
from matplotlib import pyplot as plt
import matching
import harris
import warnings
import Segmentation
import FACE_DETECTION
import FaceRecognition as FR
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")
################################## Page Layouts ###########################################################
st.set_page_config(
    page_title="Filtering and Edge detection",
    page_icon="✅",
    layout="wide",
)
################################## Page construction #######################################################
st.title("Filtering and Ege detection")
css = """
.uploadedFiles {
    display: none;
}
"""
with open(r"style.css") as design:
    st.markdown(f"<style>{design.read()}</style>", unsafe_allow_html=True)
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

#############################################################################################################
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
chosen_id = stx.tab_bar(data=[
    stx.TabBarItemData(id="tab1", title="Filters", description=''),
    stx.TabBarItemData(id="tab2", title="Histograms", description=''),
    stx.TabBarItemData(id='tab3', title='Hybrid', description=''),
    stx.TabBarItemData(id='tab4', title='Hough', description=''),
    stx.TabBarItemData(id='tab5', title='Active contour', description=''),
    stx.TabBarItemData(id='tab6', title='Harris', description=''),
    stx.TabBarItemData(id='tab7', title='SIFT', description=''),
    stx.TabBarItemData(id='tab8', title='Thresholding', description=''),
    stx.TabBarItemData(id='tab9', title='Segmentation', description=''),
    stx.TabBarItemData(id='tab10', title='Face Detection', description='')])
sidebar = st.sidebar.container()

#############################################################################################################
def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


#############################################################################################################
if chosen_id == "tab1":
    # my_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    # Add noise
    selected_noise = sidebar.selectbox('Add Noise', ('Uniform Noise', 'Gaussian Noise', 'Salt & Pepper Noise'))
    snr_value = sidebar.slider('SNR ratio', 0, step=1, max_value=100, value=50)
    sigma_noise = sidebar.slider('Sigma', 0.0, step=0.01, max_value=1.0, value=0.128)

    # Apply filter
    selected_filter = sidebar.selectbox('Apply Filter', ('Average Filter', 'Gaussian Filter', 'Median Filter'))
    mask_sizes = ['3x3', '5x5', '7x7', '9x9']
    mask_slider = sidebar.select_slider('Mask Size', options=mask_sizes)
    # sigma_slider = sidebar.slider('Sigma', 0, step=1, max_value=100, value=50, label_visibility='visible')

    # Detect edges
    edge_types = ['Sobel', 'Roberts', 'Prewitt', 'Canny Edge']
    edge = sidebar.selectbox('Detect Edges', edge_types)
    i_image, n_image, f_image, e_image = st.columns(4)
    # Load image
    if my_upload is not None:
        image = Image.open(my_upload).convert("L")
        img = np.array(image)

        # Display input image
 

        with i_image:
            st.markdown('<p style="text-align: center;">Input Image</p>', unsafe_allow_html=True)
            st.image(image, width=190)

        # Add noise
        with n_image:
            st.markdown(f'<p style="text-align: center;">Noisy Image ({selected_noise})</p>', unsafe_allow_html=True)
            if selected_noise == "Uniform Noise":
                noisy_image = fs.add_uniform_noise(img, a=0, b=sigma_noise)
            elif selected_noise == "Gaussian Noise":
                var = np.var(image) / (10 ** (snr_value / 10))
                noisy_image = fs.add_gaussian_noise(img, mean=0, var=var)
            else:
                noisy_image = fs.add_salt_pepper_noise(img, pepper_amount=sigma_noise)
            st.image(noisy_image, width=190)

        # Apply filter
        with f_image:
            st.markdown(f'<p style="text-align: center;">Filtered Image ({selected_filter})</p>', unsafe_allow_html=True)
            if selected_filter == "Gaussian Filter":
                g_filter = fs.gaussian_filter(noisy_image * 255)
                g_filter_norm = g_filter / 255.0
                st.image(g_filter_norm, width=190)
            elif selected_filter == "Average Filter":
                avg_filter = fs.average_filter(noisy_image * 255)
                st.image(avg_filter, width=190)
            else:
                removed_noise = fs.median_filter(noisy_image * 255, int(mask_slider[0]))
                removed_noise_norm = removed_noise / 255.0
                st.image(removed_noise_norm, width=190)

        # Detect edges
        with e_image:
            st.markdown(f'<p style="text-align: center;">Edge Detection Image ({edge})</p>', unsafe_allow_html=True)
            if edge == "Sobel":
                edge_img = fs.edge_detection(img, 'sobel')
            elif edge == "Roberts":
                edge_img = fs.edge_detection(img, "roberts")
            elif edge == "Prewitt":
                    edge_img = fs.edge_detection(img, "prewitt")
                    
            else:
                    edge_img = fs.edge_detection(img)
            st.image(edge_img, width=190)
#############################################################################################################
elif chosen_id == "tab2":
    histogram = sidebar.selectbox('Histogram',
                                  ('equalized image', 'normalized image',  'gray image', 'global thresholding', 'local thresholding'))

    if my_upload is not None:
        image = Image.open(my_upload)
        i_image, f_image = st.columns([1, 1])
        chart1, chart2 = st.columns([1, 1])
        converted_img = np.array(image)
        gray_scale = Histogram.rgb_to_gray(converted_img)

        with i_image:
            st.markdown(
                '<p style= "text-align: center;">Input Image</p>', unsafe_allow_html=True)
            st.image(image, width=300)

        if histogram == 'equalized image':
            radio_button = sidebar.radio(
                "", ["Histogram", "Distribution", "CDF"], horizontal=False)
            equlized_img, bins = Histogram.equalize_histogram(
                source=converted_img, bins_num=256)

            with f_image:
                st.markdown(
                    '<p style="text-align: center;">Output Image</p>', unsafe_allow_html=True)
                st.image(equlized_img, width=300)

            if radio_button == "Histogram":
                with chart1:
                    st.markdown(
                        '<p style="text-align: center;">Original Histogram</p>', unsafe_allow_html=True)
                    Histogram.hist_bar(converted_img)

                with chart2:
                    st.markdown(
                        '<p style="text-align: center;">Equalize Histogram</p>', unsafe_allow_html=True)
                    Histogram.hist_bar(equlized_img)
            elif radio_button == "Distribution":
                with chart1:
                    st.markdown(
                        '<p style="text-align: center;">Original Histogram</p>', unsafe_allow_html=True)
                    Histogram.draw_rgb_histogram(source=converted_img)

                with chart2:
                    st.markdown(
                        '<p style="text-align: center;">Equalize Histogram</p>', unsafe_allow_html=True)
                    Histogram.draw_rgb_histogram(source=equlized_img)
            elif radio_button == "CDF":
                with chart1:
                    st.markdown(
                        '<p style="text-align: center;">Original Histogram</p>', unsafe_allow_html=True)
                    Histogram.rgb_distribution_curve(source=converted_img)

                with chart2:
                    st.markdown(
                        '<p style="text-align: center;">Equalize Histogram</p>', unsafe_allow_html=True)
                    Histogram.rgb_distribution_curve(source=equlized_img)

        elif histogram == 'normalized image':
            normalized_image, norm_hist, bins = Histogram.normalize_histogram(
                source=converted_img, bins_num=256)

            with f_image:
                st.markdown(
                    '<p style="text-align: center;">Output Image</p>', unsafe_allow_html=True)
                st.image(normalized_image, width=300)

            with chart1:
                st.markdown(
                    '<p style="text-align: center;">Original Histogram</p>', unsafe_allow_html=True)
                Histogram.draw_rgb_histogram(source=converted_img)
            with chart2:

                st.markdown(
                    '<p style="text-align: center;">Normalize Histogram</p>', unsafe_allow_html=True)
                Histogram.draw_rgb_histogram(source=normalized_image)

        elif histogram == 'gray image':
            gray_scale = Histogram.rgb_to_gray(converted_img)
            radio_button = sidebar.radio(
                "", ["Histogram", "Distribution"], horizontal=False)

            with f_image:
                st.markdown(
                    '<p style="text-align: center;">Output Image</p>', unsafe_allow_html=True)
                st.image(gray_scale, width=300)
            if radio_button == "Histogram":
                with chart1:
                    st.markdown(
                        '<p style="text-align: center;">Original Histogram</p>', unsafe_allow_html=True)
                    Histogram.hist_bar(converted_img)

                with chart2:
                    st.markdown(
                        '<p style="text-align: center;">gray  Histogram</p>', unsafe_allow_html=True)
                    figure, axis = plt.subplots()
                    _ = plt.hist(converted_img.ravel(), 256)
                    st.pyplot(figure)
            elif radio_button == "Distribution":
                with chart1:
                    st.markdown(
                        '<p style="text-align: center;">Original Histogram</p>', unsafe_allow_html=True)
                    Histogram.draw_rgb_histogram(source=converted_img)

                with chart2:
                    st.markdown(
                        '<p style="text-align: center;">gray  Histogram</p>', unsafe_allow_html=True)
                    Histogram.draw_gray_histogram(
                        source=gray_scale, bins_num=256)

        elif histogram == 'global thresholding':
            slider = st.sidebar.slider(
                'Adjust the intensity', 0, 255, 128, step=1)
            global_threshold = Histogram.global_threshold(
                source=gray_scale, threshold=slider)

            with f_image:
                st.markdown(
                    '<p style="text-align: center;">Output Image</p>', unsafe_allow_html=True)
                st.image(global_threshold, width=300)

            with chart1:
                st.markdown(
                    '<p style="text-align: center;">Original Histogram</p>', unsafe_allow_html=True)
                Histogram.draw_rgb_histogram(source=converted_img)

            with chart2:
                st.markdown(
                    '<p style="text-align: center;">Global Histogram</p>', unsafe_allow_html=True)
                hist_glob, bins = Histogram.histogram(
                    source=global_threshold, bins_num=2)
                Histogram.display_bar_graph(
                    x=bins, height=[hist_glob[0], hist_glob[-1]], width=0.2)

        elif histogram == 'local thresholding':
            local_threshold = Histogram.local_threshold1(
                source=gray_scale, divs=250)

            with f_image:
                st.markdown(
                    '<p style="text-align: center;">Output Image</p>', unsafe_allow_html=True)
                st.image(local_threshold, width=300)

            with chart1:
                st.markdown(
                    '<p style="text-align: center;">Original Histogram</p>', unsafe_allow_html=True)
                Histogram.draw_rgb_histogram(source=converted_img)

            with chart2:
                st.markdown(
                    '<p style="text-align: center;">Global Histogram</p>', unsafe_allow_html=True)
                hist, bins = Histogram.histogram(
                    source=local_threshold, bins_num=2)
                Histogram.display_bar_graph(
                    x=bins, height=[hist[0], hist[-1]], width=0.2)

#############################################################################################################
elif chosen_id=='tab3': 
            # my_upload = st.sidebar.file_uploader("Upload first image", type=["png", "jpg", "jpeg"])
            second = st.sidebar.file_uploader("Upload second image", type=["png", "jpg", "jpeg"])
            High_pass_first = st.checkbox('high pass for the first image')
            flag_1 = 0
            flag_2=1
            if High_pass_first:
                flag_1 = 1
                flag_2=0

            l_image, l_1_image, r_image, r_1_image = st.columns(4)
            if my_upload  is not None:
                path_1='images/'+ my_upload.name
                with l_image:
                            st.markdown('<p style="text-align: center;">Input1 Image</p>',unsafe_allow_html=True)
                            st.image(freq.prepare(path_1))       
                with  l_1_image:
                                st.markdown('<p style="text-align: center;"> Image _1 after filtering</p>',unsafe_allow_html=True)
                                updated_path_1 = freq.getfilter(path_1,flag_1)
                                st.image(updated_path_1 ) 
            if second is not None:
                    path_2='images/'+second.name

                    with r_image:
                         st.markdown('<p style="text-align: center;">Input2 Image</p>',unsafe_allow_html=True)
                         st.image(freq.prepare(path_2))        
                    with r_1_image:
                        st.markdown('<p style="text-align: center;">Image_2 after filtering</p>',unsafe_allow_html=True)
                        updated_path_2 = freq.getfilter(path_2,flag_2)
                        st.image(updated_path_2)
                
                    c_image ,c_2,c_3= st.columns(3)
                    with c_2:
                        st.markdown('<p style="text-align: center;">Hybrid Image</p>',unsafe_allow_html=True)
                        st.image(freq.hybrid_images(updated_path_1,updated_path_2))

#############################################################################################################
elif chosen_id == "tab4":

    # Apply filter
    selected_Hough = sidebar.selectbox('Apply Hough Transform ', ('lines', 'circles', 'ellipse'))

    input_image, output_image = st.columns(2)
    # Load image
    if my_upload is not None:
        path='images/'+ my_upload.name
        image = Image.open(my_upload).convert("L")
        image = cv2.imread(path)
        img = np.copy(image)

        # Display input image
        if selected_Hough == "lines":
            hough_image = Hough.hough_lines(img, num_peaks=10)
        elif selected_Hough == "circles":
                hough_image = Hough.hough_circles(img, min_radius=20,
                                                max_radius=50)
        elif selected_Hough == "ellipse":
                hough_image = Hough.hough_circles(img, min_radius=1,
                                                max_radius=10)
        with input_image:
            st.markdown('<p style="text-align: center;">Input Image</p>', unsafe_allow_html=True)
            st.image(image, width=190)



        # Add noise
        with output_image:

            st.markdown(f'<p style="text-align: center;">Noisy Image ({selected_Hough})</p>', unsafe_allow_html=True)
            st.image(hough_image, width=190)


#############################################################################################################
elif chosen_id == "tab5":
   


    input_image, output_image = st.columns(2)
    image = data.astronaut()
    img = rgb2gray(image) 

    s = np.linspace(0, 2*np.pi, 400)
    r = 100 + 100*np.sin(s)
    c = 220 + 100*np.cos(s)
    init = np.array([r, c]).T



    # Display input image
    snake = contour.active_contour(gaussian(img, 3, preserve_range=False),
                    init, alpha=0.015, beta=10, gamma=0.001)
    chain_code = contour.calculate_chain_code(snake)

    st.write(chain_code)
    fig, ax =plt.subplots(figsize=(7, 7))
    ax.imshow(image)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    # Save the plot as an image file
    fig.savefig('images/snake.png')
    with input_image:
        st.markdown('<p style="text-align: center;">Input Image</p>', unsafe_allow_html=True)
        st.image(image, width=190)

    # Add noise
    with output_image:

        st.markdown(f'<p style="text-align: center;">Noisy Image ()</p>', unsafe_allow_html=True)
        # image_with_countour = Contour.main()
        image_with_countour=cv2.imread('images/snake.png')
        st.image(image_with_countour, width=190)

#############################################################################################################
elif chosen_id == "tab6":
   
    l_image,  r_image = st.columns(2)
    if my_upload  is not None:
        path_1='images/'+ my_upload.name
        with l_image:
                    st.markdown('<p style="text-align: center;">Input1 Image</p>',unsafe_allow_html=True)
                    image_1 = cv2.imread(path_1)
                    st.image(image_1)   

        with r_image:
                st.markdown('<p style="text-align: center;"> Image with harris</p>',unsafe_allow_html=True)
                harriResponse  = harris.harrisCorner(image_1,0.04)
                cornerImage = harris.corner2Image(image_1, harriResponse , 0.05)
                st.image(cornerImage  )
               
#############################################################################################################

elif chosen_id == "tab7":
    second = st.sidebar.file_uploader("Upload second image", type=["png", "jpg", "jpeg"])
    l_image, m_image, r_image = st.columns(3)
    radio_button = sidebar.radio(
                "", ['SSD','NCC'], horizontal=False)
    if radio_button == 'SSD':
         method = "ssd"
    else:
         
         method ="ncc"
    if my_upload  is not None:
        path_1='images/'+ my_upload.name
        with l_image:
                    st.markdown('<p style="text-align: center;">Input1 Image</p>',unsafe_allow_html=True)
                    image_1 = cv2.imread(path_1)
                    st.image(image_1,width=190)       
    if second is not None:
        path_2='images/'+second.name

        with m_image:
                    st.markdown('<p style="text-align: center;">Input2 Image</p>',unsafe_allow_html=True)
                    image_2 = cv2.imread(path_2)
                    st.image(image_2,width=190)    

      
        with r_image:
                st.markdown('<p style="text-align: center;">Output Image</p>',unsafe_allow_html=True)
                
                matched_image , match_time = matching.get_matching(path_1, path_2,method)
                st.image(matched_image,width=190)
                print(match_time)
#############################################################################################################
elif chosen_id == "tab8":
    selected_Thresholding = sidebar.selectbox('Thresholding', ('Optimal', 'otsu', 'spectral'))
    i_image,f_image, e_image = st.columns(3)
    # Load image
    if my_upload is not None:
        image = Image.open(my_upload)
        img = np.array(image)

        # Display input image
        with i_image:
            st.markdown('<p style="text-align: center;">Input Image</p>', unsafe_allow_html=True)
            st.image(image, width=200)

        # Apply Thresholding
        with f_image:
            st.markdown(f'<p style="text-align: center;"> Global {selected_Thresholding}</p>', unsafe_allow_html=True)
            if selected_Thresholding == "Optimal":
                global_img = Thresholding.optimal(img)
            elif selected_Thresholding == "otsu":
                global_img = Thresholding.otsu(img)
            elif selected_Thresholding == "spectral":
                global_img = Thresholding.spectral(img)
            cv2.imwrite('Global.jpg', global_img)
            st.image('Global.jpg', width=200)

        # Apply Thresholding
        with e_image:
            st.markdown(f'<p style="text-align: center;"> Local {selected_Thresholding}</p>', unsafe_allow_html=True)
            if selected_Thresholding == "Optimal":
                local_img = Thresholding.LocalThresholding(img , 4 ,Thresholding.optimal)
            elif selected_Thresholding == "otsu":
                local_img = Thresholding.LocalThresholding(img , 4 ,Thresholding.otsu)
            elif selected_Thresholding == "spectral":
                local_img = Thresholding.LocalThresholding(img , 4 ,Thresholding.spectral)
            cv2.imwrite('Local.jpg', local_img)
            st.image('Local.jpg', width=200)
#############################################################################################################
elif chosen_id == "tab9":
    l_image, r_image = st.columns(2)
    radio_button = sidebar.radio(
                "", ['K_means_Method','Region_Growing_method','Agglomerative','Mean shift','LUV'], horizontal=False)
    flag = True
    if radio_button == 'K_means_Method':
         method = "km"
         flag = False
         
    elif radio_button == 'Region_Growing_Method':
         method ="rg"

    elif radio_button =='Agglomerative':
         method = "ag"
    elif radio_button =='Mean shift':
         method = "ms"
    else:
         method ="luv"

    k_value = sidebar.slider('K Value', 0, step=1, max_value=10, value=4, disabled= flag )    
    if my_upload  is not None:
        image = Image.open(my_upload)
        image_1  = np.array(image)
        with l_image:
                    st.markdown('<p style="text-align: center;">Input Image</p>',unsafe_allow_html=True)
                    
                    # image_1 = cv2.cvtColor(image_1,cv2.COLOR_BGR2RGB)  
                    st.image(image_1,width=450)       
      
        with r_image:

                st.markdown('<p style="text-align: center;">Output Image</p>',unsafe_allow_html=True)
                if method == "km":
                    # image_1 = cv2.resize(image_1, (128, 128))
                    pixel_values = image_1.reshape((-1, 3))
                    pixel_values = np.float32(pixel_values)
                    k = km.KMeans(K= k_value, max_iters=100)  
                    labels = k.predict(pixel_values)
                    centers = k.cent()
                    segmented_image = centers[labels.flatten()]
                    segmented_image_rg = segmented_image.reshape(image_1.shape)
                    
                elif method == "rg":
                    # Apply region growing algorithm
                    segmented_image_rg = rg.apply_region_growing(image_1)

                elif method == "ag":
                    resized_image = cv2.resize(image_1, (128, 128))
                    segmented_image_rg = Segmentation.apply_agglomerative_clustering(resized_image,15,30)
                        
                elif method == "ms":
                    resized_image = cv2.resize(image_1, (200, 200))
                    segmented_image_rg = Segmentation.mean_shift(resized_image)
                elif method == "luv":
                    segmented_image_rg = luv.BGR_To_LUV(image_1) 

                st.image(segmented_image_rg,width=450)
   
#############################################################################################################
else:
    sidebar.empty()
