import pandas as pd
import numpy as np
import joblib
import json
import os
import time  # for measuring inference time

class RiskDetector:
    def __init__(self, model_dir="models"):
        # Load both stage models
        self.stage1_model = joblib.load(os.path.join(model_dir, "potential_risk_lightgbm_stage1.pkl"))
        self.stage2_model = joblib.load(os.path.join(model_dir, "potential_risk_xgboost_stage2.pkl"))

        # Load label encoders (if any)
        self.label_encoders = joblib.load(os.path.join(model_dir, "label_encoders.pkl"))

        # Load feature list
        with open(os.path.join(model_dir, "feature_list.json"), "r") as f:
            self.feature_list = json.load(f)

        # Load config (contains thresholds, tiers, etc.)
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            self.config = json.load(f)

        self.stage1_threshold = self.config.get("stage1_threshold", 0.6)
        self.stage2_thresholds = self.config.get("risk_thresholds", {"low": 0.4, "medium": 0.7})

        # Stage 1 feature list (for LightGBM)
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

    def preprocess_input(self, input_data: dict):
        """Convert user input into model-ready dataframe"""
        df = pd.DataFrame([input_data])

        # Encode categorical fields if label encoders exist
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                df[col] = df[col].map(lambda x: encoder.get(x, -1))

        # Ensure all features exist
        for feature in self.feature_list:
            if feature not in df.columns:
                df[feature] = 0

        return df[self.feature_list]

    def predict_two_stage_single(self, input_data: dict):
        """Run two-stage prediction pipeline and measure inference time"""
        # Preprocess the input
        X = self.preprocess_input(input_data)

        # Stage 1: LightGBM (coarse detection)
        X_stage1 = X[self.stage1_features]
        start_time = time.time()
        stage1_prob = self.stage1_model.predict_proba(X_stage1)[:, 1][0]
        stage1_time = (time.time() - start_time) * 1000  # milliseconds

        if stage1_prob < self.stage1_threshold:
            return {
                "probability": float(stage1_prob),
                "risk_tier": "Low",
                "stage_triggered": False,
                "stage1_time_ms": round(stage1_time, 3),
                "stage2_time_ms": None
            }

        # Stage 2: XGBoost (refined analysis)
        start_time = time.time()
        stage2_prob = self.stage2_model.predict_proba(X)[:, 1][0]
        stage2_time = (time.time() - start_time) * 1000  # milliseconds

        # Determine tier
        if stage2_prob < self.stage2_thresholds["low"]:
            tier = "Low"
        elif stage2_prob < self.stage2_thresholds["medium"]:
            tier = "Medium"
        else:
            tier = "High"

        return {
            "probability": float(stage2_prob),
            "risk_tier": tier,
            "stage_triggered": True,
            "stage1_time_ms": round(stage1_time, 3),
            "stage2_time_ms": round(stage2_time, 3)
        }
