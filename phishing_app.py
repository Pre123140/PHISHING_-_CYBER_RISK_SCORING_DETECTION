import streamlit as st
import joblib
import numpy as np
import lime.lime_text
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# **Load the trained phishing detection model**
model_data = joblib.load("models/phishing_model.pkl")

# **Check if the model is stored correctly**
if isinstance(model_data, tuple) and len(model_data) == 2:
    vectorizer, classifier = model_data  # Extract vectorizer and classifier
else:
    st.error("❌ Model loading error: The model file does not contain both vectorizer and classifier.")

st.title("📩 AI-Powered Phishing Email Detection")

# **Text input for email content**
email_text = st.text_area("Enter email content:", "")

if st.button("Analyze Email"):
    if email_text.strip():
        try:
            # **Convert input to TF-IDF vector**
            input_transformed = vectorizer.transform([email_text])
            input_array = input_transformed.toarray()  # Convert sparse matrix to dense array

            # **Make Prediction**
            prediction = classifier.predict(input_array)[0]
            confidence = classifier.predict_proba(input_array).max() * 100

            # **Display Prediction**
            label = "🚨 Phishing Email Detected!" if prediction == 1 else "✅ Legitimate Email"
            st.success(f"**Prediction:** {label} (Confidence: {confidence:.2f}%)")

            # **LIME Explainability**
            st.subheader("🔍 LIME Explanation")
            explainer = lime.lime_text.LimeTextExplainer(class_names=["Legitimate", "Phishing"])
            explanation = explainer.explain_instance(
                email_text,
                lambda x: classifier.predict_proba(vectorizer.transform(x))
            )

            # Display LIME Explanation
            st.pyplot(explanation.as_pyplot_figure())

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
    else:
        st.warning("⚠️ Please enter some email content for analysis.")
