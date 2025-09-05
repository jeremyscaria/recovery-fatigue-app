import os
import joblib
import gdown
import numpy as np
import streamlit as st

# =====================
# Load Model from Google Drive
# =====================
DRIVE_FILE_ID = "1rKnhDr78HdVRjtXU5Zn47kuEUYpJeF4n"  # replace with your file id
MODEL_PATH = "health_model.pkl"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

@st.cache_resource(show_spinner=False)
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model from Google Drive…")
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    return joblib.load(MODEL_PATH)

pipeline, model = load_model_from_drive()

# =====================
# Streamlit App UI
# =====================
st.title("💪 Recovery & Fatigue Predictor")
st.markdown("Enter your daily wearable stats to get predictions for **Recovery Time** and **Fatigue Score**.")

# Collect user inputs
sleep_hours = st.number_input("Sleep Hours", min_value=3.0, max_value=12.0, step=0.1, value=7.0)
sleep_score = st.slider("Sleep Score", 0, 100, 70)
hrv = st.number_input("HRV (ms)", min_value=20, max_value=120, step=1, value=65)
training_load = st.number_input("Training Load", min_value=0, max_value=1000, step=1, value=300)

# Prediction button
if st.button("Predict"):
    X_input = np.array([[sleep_hours, sleep_score, hrv, training_load]])
    X_scaled = pipeline.transform(X_input)
    recovery_pred, fatigue_pred = model.predict(X_scaled)[0]

    st.success(f"🕒 Predicted Recovery Time: {recovery_pred:.2f} hours")
    st.warning(f"⚡ Predicted Fatigue Score: {fatigue_pred:.2f}")
