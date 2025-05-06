import streamlit as st
import joblib
import numpy as np
import lime.lime_tabular
import matplotlib.pyplot as plt

def predict_risk(cvss_score, complexity):
    """Run inference on cyber risk severity."""
    model = joblib.load("models/risk_scoring_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    complexity_encoder = joblib.load("models/complexity_encoder.pkl")
    severity_encoder = joblib.load("models/severity_encoder.pkl")
    
    # Ensure proper transformation
    complexity_encoded = complexity_encoder.transform([complexity])[0]
    input_data = np.array([[cvss_score, complexity_encoded]])
    input_scaled = scaler.transform(input_data)
    
    # Predict severity
    prediction = model.predict(input_scaled)[0]
    severity_label = severity_encoder.inverse_transform([prediction])[0]
    
    return severity_label, input_scaled

# Streamlit UI
st.title("⚠️ AI-Powered Cyber Risk Scoring")
st.write("Predict the severity of cyber vulnerabilities")

cvss_score = st.number_input("CVSS Score (0-10)", min_value=0.0, max_value=10.0, step=0.1)
complexity = st.selectbox("Complexity", ["LOW", "MEDIUM", "HIGH"])

if st.button("Predict Risk"):
    severity, input_scaled = predict_risk(cvss_score, complexity)
    st.success(f"Predicted Risk Severity: **{severity}**")

    # LIME Explanation
    st.subheader("🔍 LIME Explanation")
    model = joblib.load("models/risk_scoring_model.pkl")
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array([[0, 0], [10, 2]]),  # Covering range
        feature_names=["CVSS Score", "Complexity"],
        class_names=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        discretize_continuous=True
    )
    
    exp = explainer.explain_instance(input_scaled[0], model.predict_proba)
    fig = exp.as_pyplot_figure()
    st.pyplot(fig)
