# ⚡ Potential Risk Detection System (PAN-Based)

AI-powered **potential risk detection system** designed to identify **high-risk PANs (Primary Account Numbers)** in transaction data for payment switching environments.  
This project aims to serve as a foundation for **replacing rule-based alert systems** with intelligent, data-driven anomaly detection.

---

## 🎯 Key Features

- **🔹 Two-Stage Detection Pipeline**
  - **Stage 1 (LightGBM):** Coarse-level anomaly filtering
  - **Stage 2 (XGBoost):** Fine-grained risk scoring for high-confidence cases
- **📈 Risk Tiering:** Categorizes each PAN as **Low**, **Medium**, or **High Risk**
- **💡 Real-Time PAN Analysis:** Input individual PAN statistics and get instant predictions
- **🖥️ Streamlit Web Demo:** Clean, interactive UI for stakeholders to test predictions
- **⚙️ Model Confidence & Inference Time:** Displays model confidence and latency per prediction

---

## 🧠 Model Overview

| Stage | Model    | Purpose                     | Features Used   | Threshold                           |
| ----- | -------- | --------------------------- | --------------- | ----------------------------------- |
| 1     | LightGBM | Coarse anomaly screening    | Top 10 features | 0.6                                 |
| 2     | XGBoost  | Refined risk classification | Same top 10     | Low: <0.4, Medium: <0.7, High: ≥0.7 |

**Top 10 Features (Feature Importance)**  
`TXN_COUNT`, `UNIQUE_TERMINALS`, `AVG_TIME_BETWEEN_TXN`, `APPROVED_COUNT`,  
`UNIQUE_ISSUERS`, `RAPID_TXN_COUNT`, `ACQUIRER_ENCODED`,  
`MIN_TIME_BETWEEN_TXN`, `UNIQUE_TRAN_TYPES`, `ISSUER_ENCODED`

---

## 📊 Model Performance (Validation Results)

| Model         | Precision | Recall | F1-Score | Accuracy |
| ------------- | --------- | ------ | -------- | -------- |
| LightGBM      | 0.91      | 0.69   | 0.78     | 0.98     |
| XGBoost       | 0.55      | 0.85   | 0.67     | 0.95     |
| Random Forest | 0.89      | 0.69   | 0.78     | 0.98     |
| CatBoost      | 0.91      | 0.68   | 0.78     | 0.98     |

---

## 🚀 Quick Start

### Installation

```bash
# Clone this repository
git clone https://github.com/elwin29/potential-risk-detection.git
cd potential-risk-detection

# Install dependencies
pip install -r requirements.txt
```
