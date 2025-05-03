import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')

# Paths to datasets
PHISHING_DATA_PATH = "data/phishing_emails.csv"
RISK_DATA_PATH = "data/cyber_risk_data.csv"
PROCESSED_DATA_DIR = "data/processed/"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Text cleaning function for phishing detection
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Load and preprocess phishing dataset
def preprocess_phishing_data():
    print("🔹 Loading and processing phishing dataset...")
    df = pd.read_csv(PHISHING_DATA_PATH)
    df.dropna(inplace=True)
    df["text_combined"] = df["text_combined"].apply(clean_text)
    df.to_csv(f"{PROCESSED_DATA_DIR}phishing_processed.csv", index=False)
    print("✅ Phishing data saved!")
    return df

# Load and preprocess cyber risk dataset
def preprocess_risk_data():
    print("🔹 Loading and processing cyber risk dataset...")
    df = pd.read_csv(RISK_DATA_PATH)
    df.drop(columns=['cve_id', 'notes', 'grp'], errors='ignore', inplace=True)
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    # Fill NaN in numeric columns with 0 (or another meaningful value)
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Fill NaN in categorical columns with "Unknown"
    df.fillna("Unknown", inplace=True)

    
    # Encode categorical features
    encoder = LabelEncoder()
    for col in ['vendor_project', 'product', 'vulnerability_name', 'severity']:
        df[col] = encoder.fit_transform(df[col])
        joblib.dump(encoder, f"models/{col}_encoder.pkl")
    
    df.to_csv(f"{PROCESSED_DATA_DIR}cyber_risk_processed.csv", index=False)
    print("✅ Cyber risk data saved!")
    return df

# Run preprocessing
if __name__ == "__main__":
    phishing_df = preprocess_phishing_data()
    risk_df = preprocess_risk_data()
    print("✅ Data preprocessing complete!")