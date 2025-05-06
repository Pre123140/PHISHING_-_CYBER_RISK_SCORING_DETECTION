import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/cyber_risk_data.csv")

# Convert complexity to string
df["complexity"] = df["complexity"].astype(str)

# Ensure all required complexity levels exist
required_complexities = {"LOW", "MEDIUM", "HIGH"}
existing_complexities = set(df["complexity"].unique())
missing_complexities = required_complexities - existing_complexities

for level in missing_complexities:
    df = pd.concat([
        df,
        pd.DataFrame([{"cvss": 5.0, "complexity": level, "severity": "MEDIUM"}])
    ], ignore_index=True)

# Ensure all severity levels have at least 2 entries
required_severities = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
for level in required_severities:
    count = df[df["severity"] == level].shape[0]
    if count < 2:
        extras = pd.DataFrame([
            {"cvss": 1.0 + i, "complexity": "LOW", "severity": level}
            for i in range(2 - count)
        ])
        df = pd.concat([df, extras], ignore_index=True)

# Encode 'complexity'
complexity_encoder = LabelEncoder()
df["complexity"] = complexity_encoder.fit_transform(df["complexity"])
joblib.dump(complexity_encoder, "models/complexity_encoder.pkl")

# Define features and label
X = df[["cvss", "complexity"]]
y = df["severity"]

# Encode 'severity'
severity_encoder = LabelEncoder()
y_encoded = severity_encoder.fit_transform(y)
joblib.dump(severity_encoder, "models/severity_encoder.pkl")

# Train-test split (stratify for balanced class representation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.pkl")

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model
joblib.dump(model, "models/risk_scoring_model.pkl")

print("✅ Model Training Completed & Saved Successfully!")
