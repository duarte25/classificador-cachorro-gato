import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, UnidentifiedImageError
import numpy as np

# Título da Aplicação
st.title("Classificador de Imagens: Cachorro ou Gato 🐶🐱")
colab_link = "[Clique aqui para acessar o notebook de treinamento no Google Colab](https://colab.research.google.com/drive/1VbmZ2yqE9Ayg5oWHuDI-UBDGr1Hb5LH6?usp=sharing)"
st.markdown(f"### Treinamento do Modelo: {colab_link}")

# Função para carregar o modelo treinado
@st.cache_resource  # Cache para evitar recarregar o modelo várias vezes
def load_trained_model():
    try:
        model = load_model('modelo_cachorro_gato.h5')
        st.success("Modelo carregado com sucesso!")
        return model
    except Exception as e:
        st.error("Erro ao carregar o modelo.")
        st.error(str(e))
        return None

# Carregar o modelo
model = load_trained_model()

# Interface para upload da imagem
uploaded_file = st.file_uploader("Faça upload de uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Abrir e exibir a imagem
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagem Carregada", use_column_width=True)

        # Pré-processamento da imagem
        st.write("Processando a imagem...")

        # Redimensionando a imagem para o tamanho usado no treinamento
        image = image.resize((128, 128))

        # Convertendo para array e normalizando
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Adicionando dimensão do batch

        # Realizando a inferência
        if model:
            prediction = model.predict(image_array)[0][0]  # Obtém a saída da primeira (e única) unidade
            st.write(f"Resultado do modelo: {prediction:.2f}")

            # Classificação baseada na saída
            if prediction >= 0.5:
                st.success("Classificação: **Cachorro** 🐶")
            else:
                st.success("Classificação: **Gato** 🐱")
        else:
            st.error("Modelo não está disponível para realizar a inferência.")

    except UnidentifiedImageError:
        st.error("Erro: O arquivo enviado não é uma imagem válida.")
    except Exception as e:
        st.error("Erro durante o processamento da imagem.")
        st.error(str(e))
else:
    st.info("Por favor, faça o upload de uma imagem para classificar.")
