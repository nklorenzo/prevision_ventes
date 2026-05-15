import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# ── CONFIG PAGE ──────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom SalesTier Predictor",
    page_icon="",
    layout="centered"
)

# ── CHARGEMENT DES ARTEFACTS ─────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('model_telecom_keras.keras')
    with open('scaler_telecom_keras.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoders_telecom.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    return model, scaler, label_encoders

model, scaler, label_encoders = load_artifacts()

# ── INTERFACE ────────────────────────────────────────────────
st.title("Telecom SalesTier Predictor")
st.markdown("Prédiction du **tier de vente** d'un client basé sur ses caractéristiques.")

st.divider()

# ── FORMULAIRE ───────────────────────────────────────────────
st.subheader("Informations client")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Genre", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (mois)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

st.divider()

# ── PREDICTION ───────────────────────────────────────────────
if st.button("Prédire le SalesTier", use_container_width=True, type="primary"):

    input_dict = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
    }

    input_df = pd.DataFrame([input_dict])

    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    input_scaled = scaler.transform(input_df)

    proba = model.predict(input_scaled)[0]
    predicted_class = np.argmax(proba)

    labels = {
        0: ("Faible", "MonthlyCharges < 35$"),
        1: ("Moyen", "MonthlyCharges entre 35$ et 65$"),
        2: ("Elevé", "MonthlyCharges > 65$"),
    }

    label, description = labels[predicted_class]

    st.success(f"**SalesTier prédit : {label}**")
    st.caption(description)
