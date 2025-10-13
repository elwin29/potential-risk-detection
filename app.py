import streamlit as st
import pandas as pd
import joblib
import time

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Potential Risk Demo",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Potential Risk Detector")
st.caption("Two-stage fraud risk prediction system using LightGBM + XGBoost")

# --- Load Models ---
@st.cache_resource
def load_models():
    lightgbm_model = joblib.load("models/potential_risk_lightgbm_stage1.pkl")
    xgboost_model = joblib.load("models/potential_risk_xgboost_stage2.pkl")
    return lightgbm_model, xgboost_model

try:
    lightgbm_model, xgboost_model = load_models()
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.stop()

# --- Manual Input Section ---
st.sidebar.header("🔢 Input PAN Features (Manual Entry)")

input_data = {
    "TXN_COUNT": st.sidebar.number_input("TXN_COUNT", 0, 10000, 50),
    "UNIQUE_TERMINALS": st.sidebar.number_input("UNIQUE_TERMINALS", 0, 500, 10),
    "AVG_TIME_BETWEEN_TXN": st.sidebar.number_input("AVG_TIME_BETWEEN_TXN", 0.0, 5000.0, 300.0),
    "APPROVED_COUNT": st.sidebar.number_input("APPROVED_COUNT", 0, 10000, 45),
    "UNIQUE_ISSUERS": st.sidebar.number_input("UNIQUE_ISSUERS", 0, 50, 3),
    "RAPID_TXN_COUNT": st.sidebar.number_input("RAPID_TXN_COUNT", 0, 500, 5),
    "ACQUIRER_ENCODED": st.sidebar.number_input("ACQUIRER_ENCODED", 0, 100, 1),
    "MIN_TIME_BETWEEN_TXN": st.sidebar.number_input("MIN_TIME_BETWEEN_TXN", 0.0, 5000.0, 10.0),
    "UNIQUE_TRAN_TYPES": st.sidebar.number_input("UNIQUE_TRAN_TYPES", 0, 10, 2),
    "ISSUER_ENCODED": st.sidebar.number_input("ISSUER_ENCODED", 0, 100, 2)
}

input_df = pd.DataFrame([input_data])
st.write("### 🧾 Input Data")
st.dataframe(input_df)

# --- Prediction Button ---
if st.button("🚀 Predict Risk"):
    st.markdown("---")

    try:
        features = input_df.values

        # --- Stage 1 (LightGBM) ---
        start_time = time.time()
        stage1_prob = float(lightgbm_model.predict_proba(features)[0][1])
        stage1_time = (time.time() - start_time) * 1000

        stage1_result = {
            "stage": "Stage 1 (LightGBM)",
            "probability": stage1_prob,
            "confidence": round((1 - stage1_prob) * 100 if stage1_prob < 0.5 else stage1_prob * 100, 2),
            "inference_time": round(stage1_time, 4),
            "risk_tier": "Low" if stage1_prob < 0.3 else "Medium" if stage1_prob < 0.6 else "High",
            "stage_triggered": stage1_prob >= 0.6
        }

        # --- Stage 2 (XGBoost) if triggered ---
        stage2_result = None
        if stage1_result["stage_triggered"]:
            start_time = time.time()
            stage2_prob = float(xgboost_model.predict_proba(features)[0][1])
            stage2_time = (time.time() - start_time) * 1000

            stage2_result = {
                "stage": "Stage 2 (XGBoost)",
                "probability": stage2_prob,
                "confidence": round((1 - stage2_prob) * 100 if stage2_prob < 0.5 else stage2_prob * 100, 2),
                "inference_time": round(stage2_time, 4),
                "risk_tier": "Low" if stage2_prob < 0.3 else "Medium" if stage2_prob < 0.6 else "High",
                "stage_triggered": True
            }

        # --- Display Results ---
        st.subheader("🧠 Prediction Summary")

        final_prob = stage2_result["probability"] if stage2_result else stage1_result["probability"]
        final_tier = stage2_result["risk_tier"] if stage2_result else stage1_result["risk_tier"]
        stage_triggered = stage1_result["stage_triggered"]

        st.metric("Predicted Risk Probability", f"{final_prob:.3f}")
        st.markdown(f"**Risk Tier:** {final_tier}")
        st.markdown(f"**Stage Triggered:** {stage_triggered}")

        st.markdown("---")

        st.write(f"🕒 Stage 1 Inference Time: `{stage1_result['inference_time']}` ms")
        if stage2_result:
            st.write(f"🚀 Stage 2 Inference Time: `{stage2_result['inference_time']}` ms")

        confidence = stage2_result["confidence"] if stage2_result else stage1_result["confidence"]
        st.info(f"Model is **{confidence}% confident** this PAN is {'risky' if final_prob >= 0.6 else 'not risky'}.")

        # --- JSON Outputs ---
        st.markdown("### 📄 Full JSON Output")
        st.json(stage1_result)

        if stage2_result:
            st.markdown("### 🚀 Stage 2 Result")
            st.json(stage2_result)

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
