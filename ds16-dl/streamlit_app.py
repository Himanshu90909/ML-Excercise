
import streamlit as st
import tensorflow as tf
import keras
import numpy as np
from PIL import Image

st.set_page_config(page_title="MNIST Digit Classifier", page_icon="🔢", layout="centered")

st.title("🔢 Deep Learning Model Deployment: Streamlit App")
st.write("Upload an image of a handwritten digit (28x28) to run model inference.")

@st.cache_resource
def load_keras_model():
    # Efficient model caching across Streamlit reruns
    model = keras.models.load_model("saved_models/cnn_model.keras")
    return model

try:
    model = load_keras_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption="Uploaded Image", width=150)

    # Preprocess image to 28x28 grayscale normalized tensor
    img_resized = image.resize((28, 28))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_tensor = np.expand_dims(np.expand_dims(img_array, 0), -1)

    if st.button("Run Model Inference"):
        preds = model.predict(img_tensor, verbose=0)
        predicted_class = np.argmax(preds[0])
        confidence = np.max(preds[0])

        st.metric("Predicted Digit", value=int(predicted_class))
        st.metric("Confidence Score", value=f"{confidence*100:.2f}%")

        st.subheader("Class Probabilities Distribution")
        st.bar_chart(preds[0])
