import joblib
import pandas as pd

# Load trained phishing detection model and vectorizer
pipeline = joblib.load("models/phishing_model.pkl")
tfidf_vectorizer = pipeline[0]  # TF-IDF vectorizer
classifier = pipeline[1]  # Classifier model

# Set threshold for phishing classification (default 0.5, adjust as needed)
THRESHOLD = 0.65  # Adjust this for better phishing detection balance

def predict_phishing(email_text):
    """Runs inference with threshold tuning on a given email text."""
    
    # Convert input to TF-IDF vector
    input_transformed = tfidf_vectorizer.transform([email_text])
    
    # Predict probabilities
    phishing_prob = classifier.predict_proba(input_transformed)[0][1]  # Probability of phishing
    
    # Apply threshold tuning
    prediction = 1 if phishing_prob >= THRESHOLD else 0
    label = "🚨 Phishing Email" if prediction == 1 else "✅ Legitimate Email"
    
    return {
        "Prediction": label,
        "Confidence": f"{phishing_prob * 100:.2f}%"
    }

if __name__ == "__main__":
    email_text = input("Enter email text: ")
    result = predict_phishing(email_text)
    print(result)
