"""
Day 69: Deploy a Machine Learning Model using Streamlit Library
Topic: Streamlit app for model inference

To run: streamlit run day69_streamlit_deploy.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pickle
import streamlit as st

# --- Train and save model (in production, this would be pre-trained) ---
@st.cache_resource
def train_model():
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(iris.data, iris.target)
    return model, iris

model, iris = train_model()

# --- Streamlit UI ---
st.title("Iris Species Classifier")
st.write("Adjust the sliders to predict iris species")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.4)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.6)
petal_width = st.slider("Petal Width (cm)", 0.1, 3.0, 0.2)

features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Predict"):
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    species = iris.target_names[prediction]
    
    st.success(f"Predicted Species: **{species}**")
    st.write("Probability Distribution:")
    prob_df = pd.DataFrame({
        'Species': iris.target_names,
        'Probability': probabilities
    })
    st.bar_chart(prob_df.set_index('Species'))

# --- Show feature importance ---
st.subheader("Feature Importance")
importances = model.feature_importances_
feat_df = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)
st.bar_chart(feat_df.set_index('Feature'))

# --- Batch prediction ---
st.subheader("Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV with iris features", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if all(col in df.columns for col in iris.feature_names):
        predictions = model.predict(df[iris.feature_names])
        df['Predicted_Species'] = [iris.target_names[p] for p in predictions]
        st.write(df)
        st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")

print("Key Concepts:")
print("- Streamlit: Python web framework for ML apps, no HTML/CSS needed")
print("- st.slider/input for user interaction, st.button for actions")
print("- @st.cache_resource for model caching")
print("- Deploy: streamlit run app.py, or use Streamlit Cloud")
