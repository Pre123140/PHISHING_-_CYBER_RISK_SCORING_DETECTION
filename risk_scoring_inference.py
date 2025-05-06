import joblib
import numpy as np

def predict_risk(cvss_score, complexity):
    """ Predicts the severity of a cyber vulnerability based on CVSS score and complexity. """
    
    # Load trained model and encoders
    model = joblib.load("models/risk_scoring_model.pkl")
    complexity_encoder = joblib.load("models/complexity_encoder.pkl")
    severity_encoder = joblib.load("models/severity_encoder.pkl")
    scaler = joblib.load("models/scaler.pkl")
    
    # **Get Allowed Complexity Levels**
    allowed_complexity_levels = set(complexity_encoder.classes_)

    # **Validate Complexity Input**
    if complexity not in allowed_complexity_levels:
        raise ValueError(f"❌ Unexpected complexity level: '{complexity}'. Allowed values: {allowed_complexity_levels}")

    # **Encode Complexity**
    complexity_encoded = complexity_encoder.transform([complexity])[0]

    # **Prepare Input Data**
    input_data = np.array([[cvss_score, complexity_encoded]])
    input_scaled = scaler.transform(input_data)

    # **Make Prediction**
    predicted_label = model.predict(input_scaled)[0]
    predicted_severity = severity_encoder.inverse_transform([predicted_label])[0]
    
    return predicted_severity

if __name__ == "__main__":
    # **Example Test**
    test_cvss = 6.0
    test_complexity = "MEDIUM"  # ✅ Should now work

    try:
        result = predict_risk(test_cvss, test_complexity)
        print(f"Predicted Risk Severity: {result}")
    except ValueError as e:
        print(f"❌ Error: {e}")
