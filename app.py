import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

st.set_page_config(
    page_title="Prédicteur du niveau de vente des services de télécommunication",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('model_telecom_keras.keras')
    with open('scaler_telecom_keras.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoders_telecom.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    return model, scaler, label_encoders

model, scaler, label_encoders = load_artifacts()

st.title("Prédicteur du niveau de vente des services de télécommunication")
st.markdown("Prédiction de la catégorie tarifaire d'un client à partir de ses caractéristiques.")

st.divider()

st.subheader("Informations client")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Genre", ["Male", "Female"])
    senior_citizen_label = st.selectbox("Senior Citizen", ["Non", "Oui"])
    senior_citizen = 1 if senior_citizen_label == "Oui" else 0
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (mois)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    no_phone = phone_service == "No"
    if no_phone:
        st.selectbox("Multiple Lines", ["No phone service"], disabled=True)
        multiple_lines = "No phone service"
    else:
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])

    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    no_internet = internet_service == "No"

    if no_internet:
        st.selectbox("Online Security", ["No internet service"], disabled=True)
        st.selectbox("Online Backup", ["No internet service"], disabled=True)
        st.selectbox("Device Protection", ["No internet service"], disabled=True)
        st.selectbox("Tech Support", ["No internet service"], disabled=True)
        st.selectbox("Streaming TV", ["No internet service"], disabled=True)
        st.selectbox("Streaming Movies", ["No internet service"], disabled=True)
        online_security = online_backup = device_protection = "No internet service"
        tech_support = streaming_tv = streaming_movies = "No internet service"
    else:
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

st.divider()

if st.button("Prédire le niveau de vente", use_container_width=True, type="primary"):

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

    label = labels[predicted_class]

    st.success(f"**Niveau de vente prédit : {label}**")
    
