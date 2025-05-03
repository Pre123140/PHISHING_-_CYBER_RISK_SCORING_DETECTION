import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import os

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

print("\n🚀 Training Phishing Detection Model...")

# Load dataset
df = pd.read_csv("data/phishing_emails.csv")

# Check class balance
print("Class Distribution:\n", df["label"].value_counts())

# Preprocess text data
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X = tfidf_vectorizer.fit_transform(df["text_combined"])
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE for class balancing
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# Save both vectorizer and model as a tuple
joblib.dump((tfidf_vectorizer, model), "models/phishing_model.pkl")

# Evaluate model
y_pred = model.predict(X_test)
print("\n📊 Model Performance:\n", classification_report(y_test, y_pred))

print("✅ Model Training Completed & Saved!")
