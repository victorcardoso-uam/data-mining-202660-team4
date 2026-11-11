"""Tab 2 – Preprocessing & Correlation Matrix."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.common import FEATURE_LABELS, load_catalogue, load_json, load_model, load_report

CORR_CANDIDATES = ["a", "e", "i", "q", "ad", "condition_code", "data_arc_years", "first_obs_year",
                   "H", "log10_moid", "log10_diameter", "pha"]


@st.cache_data(show_spinner="Applying the fitted preprocessor…")
def feature_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw model inputs and their transformed version, using the preprocessor fitted inside the RF pipeline."""
    meta = load_json("metadata.json")
    features = meta["features"]["numeric"] + meta["features"]["categorical"]
    prep = load_model(meta["models"]["Random Forest"]["file"]).named_steps["prep"]
    raw = load_catalogue()[features]
    transformed = pd.DataFrame(prep.transform(raw), columns=prep.get_feature_names_out(), index=raw.index)
    return raw, transformed


def scaler_table(prep) -> pd.DataFrame:
    rows = []
    for name, step, cols in prep.transformers_:
        if name == "cat":
            continue
        imputer, scaler = step.named_steps["imputer"], step.named_steps["scaler"]
        for c, med, mu, sd in zip(cols, imputer.statistics_, scaler.mean_, scaler.scale_):
            rows.append({"feature": c, "transform": "log1p → z-score" if name == "num_log" else "z-score",
                         "imputation (train median)": med, "train mean μ": mu, "train std σ": sd})
    return pd.DataFrame(rows)


def render(df: pd.DataFrame) -> None:
    st.subheader("⚙️ Preprocessing & Correlation Matrix")
    meta = load_json("metadata.json")
    num, cat = meta["features"]["numeric"], meta["features"]["categorical"]
    prep = load_model(meta["models"]["Random Forest"]["file"]).named_steps["prep"]

    st.markdown("##### 1 · Leakage control: which columns the classifier may see")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        audit = load_report("03_preprocessing", "leakage_audit.csv")
        fig = px.bar(audit, x="PR-AUC", y="feature set", orientation="h", text_auto=".2f",
                     color="feature set", color_discrete_sequence=["#E5484D", "#3B82C4", "#F08A3C"])
        fig.add_vline(x=0.06, line_dash="dot", annotation_text="random = 0.06")
        fig.update_layout(height=260, showlegend=False, margin=dict(t=20, b=10), xaxis_range=[0, 1.08],
                          yaxis_title="", xaxis_title="Cross-validated PR-AUC (train only)")
        st.plotly_chart(fig, key="pp_leak")
    with c2:
        st.markdown(
            f"- **Kept:** {', '.join(f'`{c}`' for c in num)} + one-hot `{cat[0]}`\n"
            f"- **Forbidden:** {', '.join(f'`{c}`' for c in meta['features']['forbidden'])}\n\n"
            "`pha` is *defined* as MOID ≤ 0.05 AU and H ≤ 22: with those two columns a forest gets "
            "PR-AUC = 1.00, which is the rule, not learning. Removing them is what makes the model useful "
            "for objects whose MOID or size is still unknown.")

    st.markdown("##### 2 · Missing values and their treatment")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        miss = (pd.Series(meta["missing_share_raw"]).mul(100).sort_values().rename("missing %")
                .rename_axis("column").reset_index())
        miss["status"] = np.where(miss["column"].isin(num + cat), "model feature", "excluded")
        fig = px.bar(miss, x="missing %", y="column", color="status", orientation="h", text_auto=".1f",
                     color_discrete_map={"model feature": "#3B82C4", "excluded": "#7D8597"})
        fig.update_layout(height=320, margin=dict(t=20, b=10), legend_title_text="", yaxis_title="")
        st.plotly_chart(fig, key="pp_missing")
    with c2:
        st.markdown(
            f"Raw catalogue: **{meta['raw_shape'][0]:,} rows × {meta['raw_shape'][1]} columns**.\n\n"
            "- Numeric gaps → `SimpleImputer(strategy='median')`, medians learned on the **training split only**.\n"
            "- `albedo` (97% missing) and `rot_per` (95%) are dropped, not imputed.\n"
            "- Unknown categories at inference → `OneHotEncoder(handle_unknown='ignore')`.\n"
            "- Everything lives inside a scikit-learn `Pipeline`, serialized with the model: the app never "
            "re-implements the preprocessing.")

    st.markdown("##### 3 · Scaling and encoding (fitted `ColumnTransformer` loaded from the .joblib)")
    st.dataframe(scaler_table(prep), hide_index=True,
                 column_config={k: st.column_config.NumberColumn(format="%.3f")
                                for k in ["imputation (train median)", "train mean μ", "train std σ"]})
    raw, transformed = feature_matrix()
    c1, c2 = st.columns([1, 3])
    feature = c1.selectbox("Feature", num, format_func=lambda c: FEATURE_LABELS.get(c, c), key="pp_feature")
    c1.caption("Skewed features (`a`, `ad`, observation arc) get `log1p` before the z-score, so the model and "
               "the neural network see comparable, roughly symmetric scales.")
    fig = make_subplots(rows=1, cols=2, subplot_titles=[f"raw {feature}", f"after the pipeline (z-score)"])
    fig.add_trace(go.Histogram(x=raw[feature], name="raw", nbinsx=80, marker_color="#7D8597"), row=1, col=1)
    fig.add_trace(go.Histogram(x=transformed[feature], name="transformed", nbinsx=80, marker_color="#F08A3C"),
                  row=1, col=2)
    fig.update_layout(height=300, margin=dict(t=40, b=10), showlegend=False)
    c2.plotly_chart(fig, key="pp_scaling")

    with st.expander(f"Model-ready feature matrix: {transformed.shape[0]:,} × {transformed.shape[1]} "
                     f"(including {sum(c.startswith('class_') for c in transformed.columns)} one-hot columns)"):
        st.dataframe(transformed.head(200).round(3), height=260)

    st.markdown("##### 4 · Correlation matrix")
    corr_df = df.assign(log10_moid=np.log10(df["moid_au"] + 1e-4), log10_diameter=np.log10(df["diameter_km"]),
                        pha=df["pha"].astype(int))
    c1, c2 = st.columns([3, 1])
    cols = c1.multiselect("Variables", CORR_CANDIDATES, default=CORR_CANDIDATES, key="pp_corr_cols")
    method = c2.radio("Method", ["pearson", "spearman"], horizontal=True, key="pp_corr_method")
    if len(cols) >= 2:
        corr = corr_df[cols].corr(method=method)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(height=560, margin=dict(t=20, b=10))
        st.plotly_chart(fig, key="pp_corr")
        st.caption("Read-outs: `a`, `q` and `ad` are strongly correlated (one orbit, three views), and the "
                   "observation variables (`data_arc_years`, `first_obs_year`, `condition_code`) correlate with "
                   "H, i.e. with size: big asteroids were discovered first and are tracked longer. That is why "
                   "they carry most of the classifier's signal.")
