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

st.title("🧠 Two-Stage Fraud Risk Detector")
st.caption("Stage 1: XGBoost (High Recall) → Stage 2: LightGBM (High Precision)")

# --- Load Models ---
@st.cache_resource
def load_models():
    """
    Load both models:
    - Stage 1: XGBoost (casts wide net, high recall)
    - Stage 2: LightGBM (filters results, high precision)
    """
    xgboost_model = joblib.load("models/xgboost_stage1_model_v2.pkl")
    lightgbm_model = joblib.load("models/lightgbm_stage2_model_v2.pkl")
    return xgboost_model, lightgbm_model

try:
    xgboost_model, lightgbm_model = load_models()
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.stop()

# --- Feature Definition ---
# ✅ Top 10 features used by your models (based on importance)
REQUIRED_FEATURES = [
    'APPROVED_COUNT',           # 1st - 21.26%
    'AVG_TIME_BETWEEN_TXN',     # 2nd - 18.33%
    'UNIQUE_TERMINALS',         # 3rd - 15.37%
    'RAPID_TXN_COUNT',          # 4th - 8.30%
    'UNIQUE_ISSUERS',           # 5th - 4.95%
    'MIN_TIME_BETWEEN_TXN',     # 6th - 4.64%
    'MAX_TXN_PER_DAY',          # 7th - 3.42%
    'ACQUIRER_ENCODED',         # 8th - 3.16%
    'UNIQUE_TRAN_TYPES',        # 9th - 2.44%
    'DATE_SPAN_DAYS',           # 10th - 2.12%
]

# --- Sidebar: Manual Input ---
st.sidebar.header("🔢 Input PAN Features")
st.sidebar.markdown("Enter transaction features for a single PAN:")

# ✅ FIXED: Include ALL required features in input_data
input_data = {
    "APPROVED_COUNT": st.sidebar.number_input("Approved Count", 0, 10000, 45, help="Number of approved transactions"),
    "AVG_TIME_BETWEEN_TXN": st.sidebar.number_input("Avg Time Between Txn (hours)", 0.0, 5000.0, 300.0, help="Average hours between transactions"),
    "UNIQUE_TERMINALS": st.sidebar.number_input("Unique Terminals", 0, 500, 10, help="Number of different terminals used"),
    "RAPID_TXN_COUNT": st.sidebar.number_input("Rapid Transactions (<1hr)", 0, 500, 5, help="Transactions within 1 hour of each other"),
    "UNIQUE_ISSUERS": st.sidebar.number_input("Unique Issuers", 0, 50, 3, help="Number of different issuer banks"),
    "MIN_TIME_BETWEEN_TXN": st.sidebar.number_input("Min Time Between Txn (hours)", 0.0, 5000.0, 10.0, help="Minimum hours between transactions"),
    "MAX_TXN_PER_DAY": st.sidebar.number_input("Max Transactions Per Day", 0, 1000, 20, help="Maximum transactions in a single day"),
    "ACQUIRER_ENCODED": st.sidebar.number_input("Acquirer ID (encoded)", 0, 100, 1, help="Encoded acquirer bank ID"),
    "UNIQUE_TRAN_TYPES": st.sidebar.number_input("Unique Transaction Types", 0, 10, 2, help="Number of different transaction types"),
    "DATE_SPAN_DAYS": st.sidebar.number_input("Date Span (days)", 0, 1000, 30, help="Number of days between first and last transaction"),
}

# ✅ Create DataFrame with correct feature order
input_df = pd.DataFrame([input_data])[REQUIRED_FEATURES]

st.write("### 🧾 Input Data Preview")
st.dataframe(input_df, use_container_width=True)

# --- Prediction Button ---
if st.button("🚀 Run Two-Stage Prediction", use_container_width=True):
    st.markdown("---")

    try:
        features = input_df.values

        # ==========================================
        # STAGE 1: XGBoost - Cast Wide Net (High Recall)
        # ==========================================
        with st.spinner("Running Stage 1 (XGBoost)..."):
            start_time = time.time()
            stage1_prob = float(xgboost_model.predict_proba(features)[0][1])
            stage1_time = (time.time() - start_time) * 1000

        # Calculate confidence (how sure the model is)
        stage1_confidence = round(max(stage1_prob, 1 - stage1_prob) * 100, 2)

        stage1_result = {
            "stage": "Stage 1 - XGBoost Screening",
            "probability": round(stage1_prob, 4),
            "confidence": stage1_confidence,
            "inference_time_ms": round(stage1_time, 2),
            "decision": "Flagged for Stage 2" if stage1_prob >= 0.5 else "Cleared (Not Suspicious)",
            "stage2_triggered": stage1_prob >= 0.5
        }

        # Display Stage 1 Results
        st.subheader("📊 Stage 1 Results (XGBoost - Wide Net)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fraud Probability", f"{stage1_prob:.1%}")
        with col2:
            st.metric("Confidence", f"{stage1_confidence:.1f}%")
        with col3:
            st.metric("Inference Time", f"{stage1_time:.2f} ms")

        # If Stage 1 says "not suspicious", stop here
        if not stage1_result["stage2_triggered"]:
            st.success("✅ **CLEARED** - Stage 1 assessment: Not suspicious. No further analysis needed.")
            
            with st.expander("📄 View Stage 1 Details"):
                st.json(stage1_result)
            
            st.stop()

        # ==========================================
        # STAGE 2: LightGBM - Filter & Prioritize (High Precision)
        # ==========================================
        st.warning("⚠️ **FLAGGED BY STAGE 1** - Proceeding to Stage 2 for detailed analysis...")

        with st.spinner("Running Stage 2 (LightGBM)..."):
            start_time = time.time()
            stage2_prob = float(lightgbm_model.predict_proba(features)[0][1])
            stage2_time = (time.time() - start_time) * 1000

        # Calculate confidence
        stage2_confidence = round(max(stage2_prob, 1 - stage2_prob) * 100, 2)

        # Determine final risk tier based on Stage 2 (LightGBM - High Precision)
        if stage2_prob >= 0.90:
            risk_tier = "🔴 CRITICAL"
            action = "Block card immediately - Very high confidence fraud"
            color = "red"
        elif stage2_prob >= 0.85:
            risk_tier = "🟠 HIGH"
            action = "Investigate within 24 hours - High confidence fraud"
            color = "orange"
        elif stage2_prob >= 0.75:
            risk_tier = "🟡 MEDIUM"
            action = "Review within 3 days - Moderate fraud indicators"
            color = "yellow"
        elif stage1_prob >= 0.7:
            risk_tier = "🟡 MEDIUM"
            action = "Manual review - Conflicting model signals (XGBoost confident, LightGBM uncertain)"
            color = "yellow"
        else:
            risk_tier = "🟢 LOW"
            action = "Monitor only - Weak fraud indicators"
            color = "green"

        # Calculate weighted final probability (favor LightGBM's precision)
        final_prob = 0.3 * stage1_prob + 0.7 * stage2_prob

        stage2_result = {
            "stage": "Stage 2 - LightGBM Filtering",
            "probability": round(stage2_prob, 4),
            "confidence": stage2_confidence,
            "inference_time_ms": round(stage2_time, 2),
            "risk_tier": risk_tier,
            "action": action,
            "final_probability": round(final_prob, 4)
        }

        # Display Stage 2 Results
        st.markdown("---")
        st.subheader("🎯 Stage 2 Results (LightGBM - Precision Filter)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Stage 2 Probability", f"{stage2_prob:.1%}")
        with col2:
            st.metric("Final Probability", f"{final_prob:.1%}", help="Weighted: 30% XGBoost + 70% LightGBM")
        with col3:
            st.metric("Confidence", f"{stage2_confidence:.1f}%")
        with col4:
            st.metric("Inference Time", f"{stage2_time:.2f} ms")

        # Risk Tier Display
        st.markdown("---")
        st.subheader("🚨 Final Assessment")
        
        if "CRITICAL" in risk_tier:
            st.error(f"**Risk Tier:** {risk_tier}")
            st.error(f"**Recommended Action:** {action}")
        elif "HIGH" in risk_tier:
            st.warning(f"**Risk Tier:** {risk_tier}")
            st.warning(f"**Recommended Action:** {action}")
        elif "MEDIUM" in risk_tier:
            st.info(f"**Risk Tier:** {risk_tier}")
            st.info(f"**Recommended Action:** {action}")
        else:
            st.success(f"**Risk Tier:** {risk_tier}")
            st.success(f"**Recommended Action:** {action}")

        # Model Agreement Analysis
        st.markdown("---")
        st.subheader("🔍 Model Agreement Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Stage 1 (XGBoost - High Recall):**")
            st.markdown(f"- Probability: {stage1_prob:.1%}")
            st.markdown(f"- Assessment: {'Flagged as suspicious' if stage1_prob >= 0.5 else 'Cleared'}")
        
        with col2:
            st.markdown("**Stage 2 (LightGBM - High Precision):**")
            st.markdown(f"- Probability: {stage2_prob:.1%}")
            st.markdown(f"- Assessment: {risk_tier}")

        # Agreement indicator
        agreement_diff = abs(stage1_prob - stage2_prob)
        if agreement_diff < 0.1:
            st.success("✅ **Strong Agreement** - Both models have similar confidence")
        elif agreement_diff < 0.2:
            st.info("ℹ️ **Moderate Agreement** - Models generally align")
        else:
            st.warning("⚠️ **Conflicting Signals** - Models disagree significantly. Manual review recommended.")
        
        # Performance Metrics
        st.markdown("---")
        st.subheader("⚡ Performance Metrics")
        
        total_time = stage1_time + stage2_time
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Stage 1 Time", f"{stage1_time:.2f} ms")
        with col2:
            st.metric("Stage 2 Time", f"{stage2_time:.2f} ms")
        with col3:
            st.metric("Total Time", f"{total_time:.2f} ms")

        # JSON Outputs
        with st.expander("📄 View Full Stage 1 JSON Output"):
            st.json(stage1_result)

        with st.expander("📄 View Full Stage 2 JSON Output"):
            st.json(stage2_result)

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
        st.exception(e)

# --- Footer ---
st.markdown("---")
st.caption("🧠 Two-Stage Fraud Detection System | Stage 1: XGBoost (85% Recall) → Stage 2: LightGBM (91% Precision)")