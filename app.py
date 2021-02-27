import streamlit as st
from PIL import Image, ImageOps
import numpy as np


st.write("""

# Detecção de tumor cerebral utilizando técnicas de Deep Learning em imagens de Ressonância Magnética
A classificação foi baseada no dataset [Brain MRI Images for Brain Tumor Detection]
(https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection), disponível no Kaggle,
que possui 253 imagens de Ressonância Magnética (RM) do crânio, sendo 155 com tumor e 98 sem tumor.\n

- Realizar o download de qualquer imagem de RM disponível no link acima do dataset;
- Clicar no botão abaixo para selecionar a imagem e carregá-la no sistema;
- Verificar o resultado da predição (RM com tumor ou RM sem turmor).

""")


from img_classification import teachable_machine_classification

uploaded_file = st.file_uploader("Selecione a imagem de RM ...", type="jpg")
if uploaded_file is not None:
    ##image = Image.open(uploaded_file)
    image = Image.open(uploaded_file)
    st.image(image, caption='RM carregada', use_column_width=True)
    st.write("")
    ##st.write("Classificando...")
    label = teachable_machine_classification(image, 'brain_tumor_classification.h5')
    if label == 0:
        st.warning("RM cerebral com tumor.")
    else:
        st.warning("RM cerebral sem tumor.")

st.write("""

O modelo foi treinado pela ferramenta [Google Teachble Machine] (https://teachablemachine.withgoogle.com/)
utilizando TensorFlow e Keras.
A aplicação foi desenvolvida em Python com framework web [Streamlit] (https://www.streamlit.io/).
""")


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

Portfólio
--
* [Predição de Diabetes](http://dg-diabetes-prediction.herokuapp.com/)
* [Plaforma IoT](http://dg-iot.herokuapp.com)
* [Censo Escolar](http://censo-escolar.herokuapp.com/)
""")





