import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow import keras

# Cache du chargement du modÃ¨le (trÃ¨s important en production)
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model('model_telecom_keras.h5')
    
    with open('scaler_telecom_keras.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('label_encoders_telecom.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    return model, scaler, label_encoders

model, scaler, label_encoders = load_artifacts()

st.title("ðŸ”® PrÃ©diction des Ventes - Services TÃ©lÃ©com")
st.subheader("Saisissez les caractÃ©ristiques du client")

# ==================== Formulaire ====================
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("AnciennetÃ© (mois)", 0, 72, 12)
    contract = st.selectbox("Type de contrat", 
                ['Month-to-month', 'One year', 'Two year'])
    internet = st.selectbox("Service Internet", 
                ['DSL', 'Fiber optic', 'No'])
    payment = st.selectbox("MÃ©thode de paiement", 
                ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

with col2:
    monthly_charges = st.number_input("Montant mensuel ($)", min_value=0.0, value=70.0, step=0.1)
    total_charges = st.number_input("Montant total ($)", min_value=0.0, value=800.0, step=0.1)
    senior = st.radio("Client Senior ?", ["Oui", "Non"])
    partner = st.radio("A un partenaire ?", ["Oui", "Non"])
    dependents = st.radio("A des personnes Ã  charge ?", ["Oui", "Non"])

if st.button("ðŸš€ PrÃ©dire la catÃ©gorie de vente", type="primary"):
    # CrÃ©ation du DataFrame
    input_data = pd.DataFrame([{
        'tenure': tenure,
        'Contract': contract,
        'InternetService': internet,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'SeniorCitizen': 1 if senior == "Oui" else 0,
        'Partner': partner,
        'Dependents': dependents,
        # Ajoute ici toutes les autres features que ton modÃ¨le attend
    }])

    # Encodage des variables catÃ©gorielles
    for col, le in label_encoders.items():
        if col in input_data.columns:
            input_data[col] = le.transform(input_data[col].astype(str))

    # Mise Ã  l'Ã©chelle
    input_scaled = scaler.transform(input_data)

    # PrÃ©diction
    proba = model.predict(input_scaled, verbose=0)[0]
    classe = np.argmax(proba)
    
    labels = {0: 'Faible', 1: 'Moyen', 2: 'Ã‰levÃ©e'}
    
    st.success(f"**CatÃ©gorie prÃ©dite : {labels[classe]}**")
    
    # Affichage des probabilitÃ©s
    st.bar_chart({
        'Faible': proba[0],
        'Moyen': proba[1],
        'Ã‰levÃ©e': proba[2]
    })
    
    st.caption(f"ProbabilitÃ©s : Faible = {proba[0]:.1%} | Moyen = {proba[1]:.1%} | Ã‰levÃ©e = {proba[2]:.1%}")
