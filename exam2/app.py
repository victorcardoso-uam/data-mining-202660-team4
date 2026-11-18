"""Partial Exam 2 - Team 4 - Cardiac Output Triage Dashboard.

Run (from the exam2/ folder):  streamlit run app.py
Every tab loads its content from the serialized .joblib artifacts in /models
(produced by src/exam2_task1_regression.py and src/exam2_task2_ann.py).
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
FEATURES = [
    "heart_rate_bpm",
    "blood_pressure_sys_mmHg",
    "oxygen_saturation_pct",
    "respiratory_rate_bpm",
    "body_temp_c",
]
LABELS = {
    "heart_rate_bpm": "Heart rate (bpm)",
    "blood_pressure_sys_mmHg": "Systolic BP (mmHg)",
    "oxygen_saturation_pct": "SpO₂ (%)",
    "respiratory_rate_bpm": "Respiratory rate (breaths/min)",
    "body_temp_c": "Body temperature (°C)",
}

# --- Triage policy ---------------------------------------------------------
CO_PRIORITY_THRESHOLD = 9.0  # L/min, ~10th percentile of the dataset
RED_FLAGS = {  # clinical red-flag rules on raw vitals
    "Tachycardia (HR > 120)": lambda v: v["heart_rate_bpm"] > 120,
    "Bradycardia (HR < 60)": lambda v: v["heart_rate_bpm"] < 60,
    "Hypotension (SBP < 95)": lambda v: v["blood_pressure_sys_mmHg"] < 95,
    "Hypertensive (SBP > 160)": lambda v: v["blood_pressure_sys_mmHg"] > 160,
    "Hypoxemia (SpO₂ < 92%)": lambda v: v["oxygen_saturation_pct"] < 92,
    "Tachypnea (RR > 26)": lambda v: v["respiratory_rate_bpm"] > 26,
    "Fever (T > 39 °C)": lambda v: v["body_temp_c"] > 39.0,
}

st.set_page_config(page_title="Team 4 · Cardiac Output Triage", page_icon="🫀", layout="wide")


@st.cache_resource
def load_artifacts():
    return {
        "model": joblib.load(MODELS / "best_model.joblib"),
        "scaler": joblib.load(MODELS / "scaler.joblib"),
        "regression": joblib.load(MODELS / "regression_results.joblib"),
        "ann": joblib.load(MODELS / "ann_results.joblib"),
        "summary": joblib.load(MODELS / "dataset_summary.joblib"),
    }


try:
    A = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Missing artifact: {e.filename}. Run the Task 1 and Task 2 scripts first.")
    st.stop()

st.title("🫀 Team 4 · Cardiac Output Prediction & Triage")
st.caption("Partial Exam 2 · Dataset: exam2_team4_health.csv · Target: cardiac_output_l_min")

with st.sidebar:
    st.header("Loaded artifacts")
    for f in ["best_model.joblib", "scaler.joblib", "regression_results.joblib",
              "ann_results.joblib", "dataset_summary.joblib"]:
        st.success(f"✅ models/{f}")
    st.write(f"**ANN:** `{A['model'].__class__.__name__}` "
             f"{A['model'].hidden_layer_sizes}, {A['model'].activation}")
    st.write(f"**Scaler:** `{A['scaler'].__class__.__name__}`")

tab1, tab2, tab3, tab4 = st.tabs([
    "1 · Dataset Overview",
    "2 · Regression Comparison",
    "3 · ANN Training",
    "4 · What-If Triage",
])

# --- Tab 1 -----------------------------------------------------------------
with tab1:
    S = A["summary"]
    st.subheader("Dataset overview")
    st.caption("Source: models/dataset_summary.joblib")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", S["n_rows"])
    c2.metric("Features", len(FEATURES))
    c3.metric("Target capped at 10.0 L/min", f"{S['capped_share']:.0%}")
    st.warning("Most target values sit exactly at 10.0 L/min, so the target looks capped "
               "(right-censored). This limits linear models and causes the diagonal band "
               "in the residual plots.")
    st.dataframe(S["describe"].round(2), width="stretch")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Correlation with target**")
        st.bar_chart(S["corr"]["cardiac_output_l_min"].drop("cardiac_output_l_min"))
    with col_b:
        st.markdown("**Target distribution**")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(S["target_values"], bins=30, color="tab:red", alpha=0.75)
        ax.set_xlabel("cardiac_output_l_min")
        st.pyplot(fig)
    st.markdown("**Sample rows**")
    st.dataframe(S["head"], width="stretch")

# --- Tab 2 -----------------------------------------------------------------
with tab2:
    R = A["regression"]
    st.subheader("Regression model comparison (20% holdout)")
    st.caption("Source: models/regression_results.joblib")
    m = R["metrics"].copy()
    ann_m = A["ann"]["metrics"]
    m.loc[ann_m["model"]] = [len(FEATURES), ann_m["R2"], ann_m["Adj_R2"], ann_m["MAE"], ann_m["RMSE"]]
    st.dataframe(m.style.format(precision=4).highlight_min(subset=["MAE", "RMSE"], color="#c8f7c5")
                 .highlight_max(subset=["R2", "Adj_R2"], color="#c8f7c5"), width="stretch")
    st.info(f"Best regression model by test RMSE: **{R['best_model']}**")
    choice = st.selectbox("Residual diagnostic dashboard", list(R["figures"].keys()))
    st.image(str(ROOT / R["figures"][choice]), width="stretch")
    st.markdown("**Residual health checks (train residuals)**")
    st.dataframe(R["diagnostics"], width="stretch")

# --- Tab 3 -----------------------------------------------------------------
with tab3:
    N = A["ann"]
    st.subheader("ANN · MLPRegressor (64, 32), ReLU, early stopping")
    st.caption("Source: models/ann_results.joblib + models/best_model.joblib")
    mm = N["metrics"]
    c = st.columns(5)
    c[0].metric("Test R²", f"{mm['R2']:.3f}")
    c[1].metric("Adj R²", f"{mm['Adj_R2']:.3f}")
    c[2].metric("MAE", f"{mm['MAE']:.3f}")
    c[3].metric("RMSE", f"{mm['RMSE']:.3f}")
    c[4].metric("Epochs", mm["epochs"])
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Training loss convergence (`loss_curve_`)**")
        st.line_chart(pd.DataFrame({"training loss": N["loss_curve"]}))
    with col_b:
        st.markdown("**Validation R² (early stopping monitor)**")
        st.line_chart(pd.DataFrame({"validation R²": np.clip(N["validation_scores"], 0, 1)}))
        st.caption("Clipped to [0, 1]; the first epochs start far below 0.")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(N["y_test"], N["y_pred"], alpha=0.6)
    lo, hi = min(N["y_test"].min(), N["y_pred"].min()), max(N["y_test"].max(), N["y_pred"].max())
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set(xlabel="Actual", ylabel="Predicted", title="ANN: actual vs predicted (test)")
    st.pyplot(fig)

# --- Tab 4 -----------------------------------------------------------------
with tab4:
    st.subheader("Real-time what-if scenario · Priority triage alert")
    st.caption("Live inference with models/best_model.joblib + models/scaler.joblib")
    ranges = A["summary"]["feature_ranges"]
    col_in, col_out = st.columns([1, 1.2])
    vitals = {}
    with col_in:
        for f in FEATURES:
            lo, hi, med = ranges[f]
            pad = (hi - lo) * 0.15
            hi_lim = 100.0 if f == "oxygen_saturation_pct" else round(hi + pad, 1)
            vitals[f] = st.slider(LABELS[f], float(round(lo - pad, 1)), float(hi_lim),
                                  float(med), step=0.1, key=f)
    X = pd.DataFrame([vitals])[FEATURES]
    raw_pred = float(A["model"].predict(A["scaler"].transform(X))[0])
    pred = min(raw_pred, 10.0)  # the monitor reads at most 10.0 L/min
    flags = [name for name, rule in RED_FLAGS.items() if rule(vitals)]
    low_co = pred < CO_PRIORITY_THRESHOLD
    high_risk = low_co or len(flags) >= 2

    with col_out:
        st.metric("Predicted cardiac output", f"{pred:.2f} L/min",
                  delta=f"{pred - CO_PRIORITY_THRESHOLD:+.2f} vs {CO_PRIORITY_THRESHOLD} threshold")
        if high_risk:
            st.error("🚨 **PRIORITY TRIAGE ALERT**: high-risk patient, escalate immediately.")
        else:
            st.success("✅ **NORMAL**: standard triage queue.")
        st.markdown("**Policy:** priority alert if predicted CO < "
                    f"{CO_PRIORITY_THRESHOLD} L/min (≈ bottom 10% of patients) "
                    "**or** ≥ 2 clinical red flags.")
        st.markdown(f"- Low predicted CO: {'🔴 yes' if low_co else '🟢 no'}")
        st.markdown(f"- Red flags ({len(flags)}): " + (", ".join(flags) if flags else "none"))
        st.caption(f"Raw ANN output: {raw_pred:.3f} L/min")
