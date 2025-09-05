import streamlit as st
import joblib
import numpy as np

# Load pipeline + model
pipeline, model = joblib.load("health_model.pkl")

st.title("💪 Recovery & Fatigue Predictor")

st.write("Enter your daily wearable stats:")

# Input fields
sleep_hours = st.number_input("Sleep Hours", min_value=3.0, max_value=12.0, step=0.1)
sleep_score = st.slider("Sleep Score (0-100)", 0, 100, 70)
hrv = st.number_input("HRV (ms)", min_value=20, max_value=120, step=1)
training_load = st.number_input("Training Load", min_value=0, max_value=1000, step=1)

# Predict button
if st.button("Predict"):
    X_input = np.array([[sleep_hours, sleep_score, hrv, training_load]])
    X_scaled = pipeline.transform(X_input)
    pred = model.predict(X_scaled)
    
    recovery_pred, fatigue_pred = pred[0]
    st.success(f"Predicted Recovery Time: {recovery_pred:.2f} hours")
    st.warning(f"Predicted Fatigue Score: {fatigue_pred:.2f}")
