import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. LOAD THE SAVED ASSETS ---
@st.cache_data
def load_assets():
    """Loads the pre-trained model, label encoder, and symptom columns."""
    model = joblib.load('random_forest.joblib')
    encoder = joblib.load('encoder.joblib')
    symptoms = joblib.load('symptom_columns.joblib')
    return model, encoder, symptoms

model, encoder, symptoms = load_assets()

# --- 2. PAGE CONFIG ---
st.set_page_config(
    page_title="MediMitra: Disease Prediction System",
    page_icon="💊",
    layout="centered"
)

# --- 3. HEADER SECTION ---
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 38px !important;
        font-weight: bold;
        color: #FFFFFF;   /* White Title */
    }
    .subtitle {
        text-align: center;
        font-size: 18px !important;
        color: #7F8C8D;
        margin-bottom: 15px;
    }
    .author {
        text-align: center;
        font-size: 16px !important;
        color: #27AE60;
        margin-top: 0px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-title">💊 MediMitra Pre-Version : Disease Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">🩺 Enter your symptoms and let MediMitra guide you towards the most likely condition.</p>', unsafe_allow_html=True)
st.markdown('<p class="author">👨‍💻 Built by: <b>Mohd Kaif</b></p>', unsafe_allow_html=True)

st.divider()

# --- 4. INPUT SECTION ---
st.subheader("🔍 Select Symptoms")
selected_symptoms = st.multiselect(
    "🤒 Choose the symptoms you are experiencing:",
    options=symptoms,
    placeholder="Type or scroll to search symptoms"
)

# --- 5. PREDICTION LOGIC ---
if st.button("🚀 Predict Disease", use_container_width=True):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom to proceed.")
    else:
        # Create input vector
        input_data = np.zeros(len(symptoms))
        for symptom in selected_symptoms:
            idx = np.where(symptoms == symptom)[0][0]
            input_data[idx] = 1
        input_data = input_data.reshape(1, -1)

        # Make prediction
        prediction_index = model.predict(input_data)[0]
        predicted_disease = encoder.inverse_transform([prediction_index])[0]

        # --- 6. RESULT DISPLAY ---
        st.success(f"✅ **Predicted Disease:** {predicted_disease}")
        st.info("💡 This prediction is for **demonstration purposes only**. Always consult a doctor for an accurate medical diagnosis.")

# --- 7. SIDEBAR INFO ---
st.sidebar.header("ℹ️ About MediMitra")
st.sidebar.write(
    """
    🤖 **MediMitra** is a smart disease prediction assistant powered by a **Random Forest model**.  

    **⚙️ Workflow:**
    - 📝 Input: Selected symptoms  
    - 🔬 ML Model: Random Forest Classifier  
    - 🏥 Output: Predicted disease  

    ⚠️ **Disclaimer:**  
    This app is **not a substitute for medical advice**.  
    Please consult a healthcare professional for accurate diagnosis.
    """
)

