import os
import pickle
import cv2 
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import opendatasets as od

from subprocess import call
import subprocess
import sys
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split

# #Multi-pages
# st.set_page_config(
#     page_title="Home",
#     page_icon='😄'
# )

#Hide "Made by Streamlit"
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
st.markdown(
    """
    <style>
    img {
        max-width: 50%;
        height: auto;
    }
    </style>
    """,
    unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .tab-content {{
        display: none;
    }}
    .tab-content.active {{
        display: block;
    }}
    </style>
    """,
    unsafe_allow_html=True)


#Main page
st.title("Fruit classification")
# st.sidebar.success("Select a page above.")

labels = {
        0: 'freshapples',
        1: 'freshbanana',
        2: 'freshoranges',
        3: 'rottenapples',
        4: 'rottenbanana',
        5: 'rottenoranges'}

#Load retrained MobileNetV2 model
model1 = keras.models.load_model('MB_V2[224x224][3].h5')
model2 = keras.models.load_model('fruit_classifier_vgg16_ver2.h5')

sidebar = st.sidebar

# Tạo container cho cột bên dưới
container = st.container()
with container:
    #Upload an image here
    uploaded_file = st.file_uploader("Choose a image file", type="jpg")
    opencv_image = []
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        img = cv2.resize(opencv_image, (224,224))
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        st.image(img, channels="BGR")

        def output(img,model,label=''):
            img = cv2.resize(img, (224,224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fig = plt.figure(figsize=(5,5),dpi=10)
            plt.imshow(img)
            img=np.expand_dims(img,[0])
            answer=model.predict(img)
            y_class = answer.argmax(axis=-1)
            y = " ".join(str(x) for x in y_class)
            y = int(y)
            res = labels[y]
            S = ['Predicted: ' + res]
            # for i in range(len(answer[0])):
            #     S.append(labels[i] + ': ' + str("{0:.2%}".format(answer[0][i])))
            if (label):
                S.append('True label: ' + label)
            return S
        # Thêm nội dung vào cột bên dưới
        col1, col2 = st.tabs(["MobileNetV2","VGG16"])
        with col1:
            st.write(output(opencv_image,model1))
        with col2:
            st.write(output(opencv_image,model2))
            st.write("---------------------")
    
   

    
