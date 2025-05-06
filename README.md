# README.md

# 🔐 AI-Powered Phishing Detection & Cyber Risk Scoring

This project combines NLP-based phishing email detection with ML-based cyber vulnerability risk scoring using CVSS metrics. Designed for real-time security decision-making, it includes class balancing (SMOTE), threshold tuning, and explainable predictions with Streamlit-based UI.

---

## 🚀 Project Highlights

- **Two Modules**:
  - **Phishing Email Detection**: Detects phishing emails using TF-IDF and Random Forest.
  - **Cyber Risk Scoring**: Predicts the severity (LOW–CRITICAL) of software vulnerabilities using CVSS score and attack complexity.

- **Key Features**:
  - Custom text preprocessing with NLTK
  - Class imbalance handling using SMOTE
  - Encoders and scaler persistence using `joblib`
  - Real-time UI with Streamlit for both phishing detection and risk prediction
  - LIME explainability (visual only)

---

## 📁 Project Structure

```
├── data/
│   ├── phishing_emails.csv
│   ├── cyber_risk_data.csv
│   └── processed/
├── models/
│   ├── phishing_model.pkl
│   ├── risk_scoring_model.pkl
│   ├── encoders & scaler.pkl
├── phishing_detection.py
├── phishing_inference.py
├── phishing_app.py
├── risk_scoring.py
├── risk_scoring_inference.py
├── risk_scoring_app.py
├── data_preprocessing.py
├── requirements.txt
```

---

## 📊 Datasets Used

- `phishing_emails.csv` – Contains combined email text and labels (0: legitimate, 1: phishing)
- `cyber_risk_data.csv` – Contains CVSS scores, complexity levels, and severity labels

---

## 🧠 Tools & Libraries

- `pandas` – Data manipulation, loading and cleaning CSVs
- `numpy` – Numeric operations and array handling
- `scikit-learn` – Random Forest models, encoding, scaling, and evaluation
- `nltk`, `re`, `string` – Text cleaning and stopword filtering
- `imblearn` – SMOTE for class balancing
- `joblib` – Saving and loading models and encoders
- `lime` – Visual explanation of model predictions (LIME plots)
- `streamlit` – Lightweight UI for deployment and interaction
- `matplotlib` – Visualizing explanations (LIME output)

---

## 🧪 How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt

```

### Step 2: Preprocess Data
```bash
python data_preprocessing.py
```

### Step 3: Train Models
```bash
python phishing_detection.py
python risk_scoring.py
```

### Step 4: Run Apps
```bash
streamlit run phishing_app.py
streamlit run risk_scoring_app.py
```

---

## 🧩 How it Works

### Phishing Detection
- TF-IDF vectorizes the combined email text
- SMOTE handles class imbalance
- Random Forest model predicts phishing label
- LIME shows visual word influence

### Cyber Risk Scoring
- Inputs: `CVSS score` (float) + `Complexity` (LOW, MEDIUM, HIGH)
- Encoded and scaled inputs are fed into a Random Forest
- Predicts risk severity: LOW, MEDIUM, HIGH, CRITICAL
- LIME visually shows feature impact

---

## 🌍 Use Cases

### Phishing Detection:
- Finance: Filter credential-stealing scam mails
- Healthcare: Block ransomware via phishing
- SaaS/EdTech: Monitor employee inboxes
- E-commerce: Detect spoofed brand emails

### Risk Scoring:
- Cloud IT: Prioritize CVEs in infrastructure
- Healthcare: Score risks in medical devices
- Government: Evaluate national-level CVSS risk
- Enterprises: Triage incidents with SLA mapping

---

## 📌 Sample Inference

### Phishing Test (CLI):
```bash
python phishing_inference.py
> Enter email text: "Click to reset your bank password."
> Prediction: 🚨 Phishing Email | Confidence: 92.7%
```

### Risk Scoring Test (CLI):
```bash
python risk_scoring_inference.py
> CVSS: 7.5, Complexity: MEDIUM
> Predicted Severity: HIGH
```

---
## 📘 Conceptual Study
Explore the theoretical foundations, modeling choices, and future scope in the accompanying PDF:
📄 [`conceptual_study_phishing_cyber_risk.pdf`](./conceptual_study_phishing_cyber_risk.pdf)

Includes:
- Business relevance and threat landscape
- ML pipeline design decisions
- Alternatives to TF-IDF and Random Forest
- Future evolution using LLMs or SHAP

  ---

## 🔭 Future Enhancements

- Integrate BERT/LLMs for advanced phishing detection
- Add real-time email scraping via IMAP
- Expand CVSS input features (impact score, access vector)
- SHAP-based enterprise explanation integration
- Auto-remediation or alert forwarding to SOC tools

---

## 🔒 Strategic Value

This project bridges the gap between human-targeted (phishing) and system-level (CVE) threats, offering a transparent, AI-driven, real-time solution for security teams. With interpretable outputs and modular components, it can serve as a foundational element in modern cybersecurity automation and decision support.

---

## 📜 License

This project is open for educational use only. For commercial deployment, contact the author.

---

## 📬 Contact
If you'd like to learn more or collaborate on projects or other initiatives, feel free to connect on [LinkedIn](https://www.linkedin.com/in/prerna-burande-99678a1bb/) or check out my [portfolio site](https://youtheleader.com/).

