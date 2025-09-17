import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. LOAD THE SAVED ASSETS ---
# We use @st.cache_data to load the model and other assets only once.
# This makes the app run faster.
@st.cache_data
def load_assets():
    """Loads the pre-trained model, label encoder, and symptom columns."""
    model = joblib.load('random_forest.joblib')
    encoder = joblib.load('encoder.joblib')
    symptoms = joblib.load('symptom_columns.joblib')
    return model, encoder, symptoms

model, encoder, symptoms = load_assets()

# --- 2. SET UP THE USER INTERFACE (UI) ---
st.set_page_config(page_title="Disease Predictor", page_icon="🩺")

st.title("🩺 Disease Prediction System")
st.write("""
Enter the symptoms you are experiencing, and this app will predict the most likely disease.
This is a demonstration model and **should not** be used for actual medical diagnosis.
""")

# Create a multi-select box for symptoms
# The list of symptoms is loaded from our saved file
selected_symptoms = st.multiselect(
    "Select your symptoms:",
    options=symptoms,
    placeholder="Choose your symptoms"
)

# --- 3. PREDICTION LOGIC ---
# Create a button to trigger the prediction
if st.button("Predict Disease", type="primary"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # Create the input array for the model
        # Start with an array of all zeros
        input_data = np.zeros(len(symptoms))

        # Set the corresponding symptom indices to 1
        for symptom in selected_symptoms:
            # Find the index of the symptom in our columns list
            idx = np.where(symptoms == symptom)[0][0]
            input_data[idx] = 1

        # Reshape the data for the model (it expects a 2D array)
        input_data = input_data.reshape(1, -1)

        # Make the prediction
        prediction_index = model.predict(input_data)[0]

        # Convert the prediction index back to the disease name
        predicted_disease = encoder.inverse_transform([prediction_index])[0]

        # --- 4. DISPLAY THE RESULT ---
        st.success(f"**Predicted Disease:** {predicted_disease}")
        st.info("Remember to consult with a healthcare professional for an accurate diagnosis.")

st.sidebar.header("About")
st.sidebar.info(
    "This app uses a Random Forest model trained on a dataset of symptoms "
    "to predict potential diseases. It demonstrates a complete machine learning workflow "
    "from data to a deployed web application."
)