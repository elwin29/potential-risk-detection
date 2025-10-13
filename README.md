# 🔍 Fraud Detection System

Machine learning-based fraud detection system for identifying anomalous PANs (Primary Account Numbers) in transaction data.

## 🎯 Features

- **Single PAN Prediction**: Real-time fraud detection for individual PANs
- **Batch Processing**: Analyze multiple PANs simultaneously
- **Risk Tiering**: 5-level risk classification (CRITICAL, HIGH, MEDIUM, LOW, NORMAL)
- **Interactive Dashboard**: User-friendly Streamlit interface
- **Demo Mode**: Test with sample data

## 📊 Model Performance

- **Precision @ 0.9**: 88.0%
- **Recall @ 0.9**: 75.0%
- **Algorithm**: XGBoost
- **Features**: 15 transaction-based features

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone <your-repo-url>
cd fraud-detection

# Install dependencies
pip install -r requirements.txt