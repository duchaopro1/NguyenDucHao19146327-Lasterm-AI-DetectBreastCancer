
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_option_menu import option_menu
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("Hue.h5") #model m train

selected = option_menu(None, ["Diagnostic", "More"], 
    icons=[ "upload", 'bookmark-fill'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "green"}
    }
)
if selected == "Diagnostic":
            ### load file
   uploaded_file = st.file_uploader("Choose an image file", type=["jpg","jpeg","png"])

   map_dict = {0: 'BENIGN',
            1: 'MALIGNANT',
            2: 'NORMAL'}
    
 
   if uploaded_file is not None:
    # Convert the file
            img = image.load_img(uploaded_file,target_size=(64,64)) #xử lí ảnh theo cách m làm
            ima = image.load_img(uploaded_file,target_size=(250,250))
            st.image(ima, channels="RGB") #hiển thị ảnh
            img = img_to_array(img)
            img = img.reshape(1,64,64,3)
            img = img.astype('float32')
            img = img/255
                
    #Button: nút dự đoán sau khi up ảnh
            Genrate_pred = st.button("Generate Prediction") 
    
            if Genrate_pred:    
                  prediction = model.predict(img).argmax()
                  st.write("**Predicted Label for the image is {}**".format(map_dict[prediction]))
                  if(prediction == 1): 
                    st.write("Bạn có một khối u không mấy hiền lành :< Nhưng mà đừng lo vì chúng ta đã phát hiện ra nó")
   
if selected == "More":
      st.title("BỆNH UNG THƯ VÚ")
