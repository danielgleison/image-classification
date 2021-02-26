import streamlit as st
from PIL import Image, ImageOps
import numpy as np


st.write("""

# Classificação de Imagens com Deep Learning
Aplicação acadêmica de detecção de tumor cerebral utilizado Tensor Flow e Keras.\n
O treinamento do modelo de classificação foi baseado no dataset [Brain MRI Images for Brain Tumor Detection]
(https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection), disponível no Kaggle,
que possui 253 imagens de Ressonância Magnética (RM), sendo 155 com tumor e 98 sem tumor.\n
Código adaptado de Kimaru Thagana, disponível no [GitHub](https://github.com/KimaruThagna/Picture-lytics).\n
Carregue a imagem de RM para predição do diagnóstico: com tumor ou sem tumor.


""")






from img_classification import teachable_machine_classification

uploaded_file = st.file_uploader("Choose a brain MRI ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, 'brain_tumor_classification.h5')
    if label == 0:
        st.write("The MRI scan has a brain tumor")
    else:
        st.write("The MRI scan is healthy")



