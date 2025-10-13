import streamlit as st
import pandas as pd
import json
import time
from risk_detector import RiskDetector

st.set_page_config(page_title="Potential Risk Detection Demo", layout="wide")

st.title("💳 Potential Risk Detection Demo")
st.markdown("This demo simulates a two-stage risk prediction system using LightGBM + XGBoost.")

# Load detector
detector = RiskDetector(model_dir="models")

st.sidebar.header("🧮 Input Features")
input_data = {}
for feature in detector.stage1_features:
    input_data[feature] = st.sidebar.number_input(f"{feature}", min_value=0.0, value=1.0, step=1.0)

if st.sidebar.button("🔍 Predict Risk"):
    with st.spinner("Running prediction..."):
        result = detector.predict_two_stage_single(input_data)
        st.success("Prediction completed!")

        # --- Main results ---
        st.subheader("🧠 Prediction Summary")

        stage1_prob = result["probability"]
        risk_tier = result["risk_tier"]
        stage_triggered = result["stage_triggered"]
        stage1_time = result.get("stage1_time_ms", 0)
        stage2_time = result.get("stage2_time_ms", None)

        # Compute confidence
        if stage_triggered:
            confidence = round(stage1_prob * 100, 2)
            conf_text = f"Model is **{confidence}% confident** this PAN is risky."
        else:
            confidence = round((1 - stage1_prob) * 100, 2)
            conf_text = f"Model is **{confidence}% confident** this PAN is *not risky*."

        # Display results
        st.metric("Predicted Risk Probability", f"{stage1_prob:.3f}")
        st.write(f"**Risk Tier:** {risk_tier}")
        st.write(f"**Stage Triggered:** {stage_triggered}")

        # Show times
        st.write("---")
        st.write(f"⏱️ Stage 1 Inference Time: `{stage1_time:.3f} ms`")
        if stage2_time is not None:
            st.write(f"⏱️ Stage 2 Inference Time: `{stage2_time:.3f} ms`")
        else:
            st.write("⚙️ Stage 2 not triggered — prediction confidently below 0.6 threshold.")

        # Confidence text
        st.info(conf_text)

        # JSON Output
        st.write("---")
        st.subheader("📄 Full JSON Output")
        st.json(result)

st.markdown("---")
st.markdown("Built with Streamlit using LightGBM & XGBoost | © 2025 Potential Risk Demo")
