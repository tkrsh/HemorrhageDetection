import streamlit as st
from streamlit_discourse import st_discourse
import os 
import pydicom 
import cv2
import numpy as np
from fastai.vision.all import * 
learner = load_learner("/home/tkrsh/Projects/Hemorrhage/model_b_1.pkl")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

import time
from PIL import Image

from streamlit_disqus import st_disqus
from streamlit_echarts import st_echarts


def return_options():
    pass
     
    
st.write(""" # Intracranial Hemorrhage Classification""")
st.write(""" ### In this Demo Version of our model we estimate whether the patient has Hemorrhage or the CT Scan is Normal """)
st.write(""" #### Generated Result will show a bar graph with probabilities of the cases""")


st.write("Source https://hemorrhage.tkrsh.com")
    
showWarningOnDirectExecution = False

@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img


image_file = st.file_uploader("Upload Image Here ", type="jpg", accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None)


if image_file is not None:
    file_details = {"FileName":image_file.name,"FileType":image_file.type}
    # st.write(file_details)
    img = load_image(image_file)
    st.image(img,caption=f"Uploaded Image :  {image_file.name}",)
    with open(os.path.join("/home/tkrsh/Projects/Hemorrhage",image_file.name),"wb") as f: 
      f.write(image_file.getbuffer())         
    prediction = learner.predict(f"/home/tkrsh/Projects/Hemorrhage/{image_file.name}")
    p_1 = round(float(prediction[-1][0]),1)
    p_2 = round(float(prediction[-1][1]),1)
     
    options = {
        "xAxis": {
            "type": "category",
            "data": ["Hemorrhage ", "Normal "]
        },
        "yAxis": {"type": "value"},
        "series": [
            {"data":[p_1,p_2] , "type": "bar"}
        ],
    }
    
    st.success("Image Uploaded To The Server")
    time.sleep(1)
    st.success("Computing Predictions Please Wait")
    time.sleep(4)
    if int(prediction[1]) == 0:
        st.write("#### Your Diagnosis: Hemorrhage is present """)
    if int(prediction[1]) == 1:
        st.write(""" #### Your Diagnosis: Scan Is Normal """)
    
    
    
    
    st_echarts(options=options, height=400, width=700)
    time.sleep(6)
    
    st.write("##### Please reach out to clear you concerns we are here to help tkrsh@tkrsh.com")
   # st_disqus("Tkrsh",url="https://tkrsh.disqus.com/")

    






