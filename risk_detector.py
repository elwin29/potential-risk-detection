import pandas as pd
import numpy as np
import joblib
import json
import os
import time

class RiskDetector:
    def __init__(self, model_dir="models"):
        """Initialize and load models, encoders, and configuration."""
        self.model_dir = model_dir

        # === Load both stage models ===
        self.stage1_model = joblib.load(os.path.join(model_dir, "potential_risk_lightgbm_stage1.pkl"))
        self.stage2_model = joblib.load(os.path.join(model_dir, "potential_risk_xgboost_stage2.pkl"))

        # === Load label encoders (optional) ===
        label_encoders_path = os.path.join(model_dir, "label_encoders.pkl")
        if os.path.exists(label_encoders_path):
            self.label_encoders = joblib.load(label_encoders_path)
        else:
            self.label_encoders = {}

        # === Load config ===
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)
        else:
            self.config = {}

        # === Load feature list if available ===
        feature_list_path = os.path.join(model_dir, "feature_list.json")
        if os.path.exists(feature_list_path):
            with open(feature_list_path, "r") as f:
                self.feature_list = json.load(f)
        else:
            self.feature_list = []

        # Threshold configuration
        self.stage1_threshold = self.config.get("stage1_threshold", 0.6)
        self.stage2_thresholds = self.config.get(
            "risk_thresholds", {"low": 0.4, "medium": 0.7}
        )

        # === Define top 10 features used by both models ===
        self.stage1_features = [
            'TXN_COUNT',
            'UNIQUE_TERMINALS',
            'AVG_TIME_BETWEEN_TXN',
            'APPROVED_COUNT',
            'UNIQUE_ISSUERS',
            'RAPID_TXN_COUNT',
            'ACQUIRER_ENCODED',
            'MIN_TIME_BETWEEN_TXN',
            'UNIQUE_TRAN_TYPES',
            'ISSUER_ENCODED'
        ]

    # ---------------------------------------------------------------
    def preprocess_input(self, input_data: dict):
        """Convert user input (dict) into a clean dataframe."""
        df = pd.DataFrame([input_data])

        # Encode categorical fields if label encoders exist
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                df[col] = df[col].map(lambda x: encoder.get(x, -1))

        # Ensure missing features exist (fill with 0)
        for feat in self.stage1_features:
            if feat not in df.columns:
                df[feat] = 0

        # Select only the expected features for model input
        df = df[self.stage1_features]
        return df

    # ---------------------------------------------------------------
    def predict_two_stage_single(self, input_data: dict):
        """Run 2-stage prediction pipeline with timing + confidence display."""
        X = self.preprocess_input(input_data)

        # === Stage 1: LightGBM ===
        start_time = time.time()
        stage1_prob = float(self.stage1_model.predict_proba(X)[:, 1][0])
        stage1_time = time.time() - start_time

        # Confidence (how sure the model is)
        stage1_confidence = round(abs(stage1_prob - 0.5) * 200, 2)  # scale to 0–100%

        # If below threshold, stop here (no need for Stage 2)
        if stage1_prob < self.stage1_threshold:
            return {
                "stage": "Stage 1 (LightGBM)",
                "probability": stage1_prob,
                "confidence": stage1_confidence,
                "inference_time": round(stage1_time, 4),
                "risk_tier": "Low",
                "stage_triggered": False
            }

        # === Stage 2: XGBoost ===
        start_time = time.time()
        stage2_prob = float(self.stage2_model.predict_proba(X)[:, 1][0])
        stage2_time = time.time() - start_time
        stage2_confidence = round(abs(stage2_prob - 0.5) * 200, 2)

        # Determine risk tier
        if stage2_prob < self.stage2_thresholds["low"]:
            tier = "Low"
        elif stage2_prob < self.stage2_thresholds["medium"]:
            tier = "Medium"
        else:
            tier = "High"

        # Return complete detailed output
        return {
            "stage": "Stage 2 (XGBoost)",
            "probability": stage2_prob,
            "confidence": stage2_confidence,
            "inference_time": round(stage1_time + stage2_time, 4),
            "risk_tier": tier,
            "stage_triggered": True
        }
