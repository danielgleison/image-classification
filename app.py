import streamlit as st
from PIL import Image, ImageOps
import numpy as np


st.write("""

# Classificação de Imagens com Deep Learning
Aplicação acadêmica de detecção de tumor cerebral utilizado Tensor Flow e Keras.\n
O treinamento do modelo de classificação foi baseado no dataset [Brain MRI Images for Brain Tumor Detection]
(https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection), disponível no Kaggle,
que possui 253 imagens de Ressonância Magnética (RM), sendo 155 com tumor e 98 sem tumor.\n

Carregue a imagem de RM para predição do diagnóstico: com tumor ou sem tumor.


""")



from img_classification import teachable_machine_classification

uploaded_file = st.file_uploader("Selecione a imagem de RM ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='RM carregada.', use_column_width=True)
    st.write("")
    st.write("Classificando...")
    label = teachable_machine_classification(image, 'brain_tumor_classification.h5')
    if label == 0:
        st.write("A RM cerebral analisada tem tumor.")
    else:
        st.write("A RM cerebral analisada não tem tumor.")


st.sidebar.write("""

Daniel Gleison M. Lira
--
Mestrando em Ciências da Computação
Universidade Estadual do Ceará
[daniel.gleison@aluno.uece.br](mailto:daniel.gleison@aluno.uece.br)
[github.com/danielgleison](https://github.com/danielgleison)
[linkedin.com/in/danielgleison] (https://www.linkedin.com/in/danielgleison/)
""")


# Portfolio
st.sidebar.write("""

Portfólio de Apps Web
--
* [Predição de Diabetes](http://dg-diabetes-prediction.herokuapp.com/)
* [Plaforma IoT](http://dg-iot.herokuapp.com)
* [Censo Escolar](http://censo-escolar.herokuapp.com/)
""")


# QR-Code
image = Image.open('qrcode.jpg')
st.sidebar.image(image, width=100, use_column_width=False, caption='v0.19.200221')


from img_classification import teachable_machine_classification



