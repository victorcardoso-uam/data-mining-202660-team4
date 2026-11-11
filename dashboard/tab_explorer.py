"""Tab 1 – Data Explorer & EDA."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.common import FEATURE_LABELS, PHA_COLORS, TIER_COLORS

HIST_FEATURES = ["a", "e", "i", "q", "ad", "H", "moid_au", "diameter_m", "condition_code",
                 "data_arc_years", "first_obs_year", "v_inf_kms", "closest_approach_LD", "pha_proba"]
LOG_FRIENDLY = {"a", "ad", "moid_au", "diameter_m", "data_arc_years", "closest_approach_LD"}
COLOR_BY = {"Official PHA flag (target)": ("pha_label", PHA_COLORS),
            "Priority tier (policy)": ("tier", TIER_COLORS),
            "Orbital family": ("class_name", None)}
SAMPLE_COLS = ["pdes", "full_name", "pha", "class_name", "H", "a", "e", "i", "q", "ad", "moid_au",
               "diameter_m", "diameter_source", "condition_code", "data_arc_years", "first_obs_year",
               "closest_approach_LD", "v_inf_kms", "tier"]


def render(df: pd.DataFrame) -> None:
    st.subheader("📊 Data Explorer & EDA")
    st.caption("JPL near-Earth asteroid catalogue (Kaggle, 2025 extraction) enriched with the close approaches "
               "2015–2035. Target: `pha`, the official *potentially hazardous asteroid* flag.")

    c1, c2, c3 = st.columns([2, 2, 1])
    families = c1.multiselect("Orbital family", sorted(df["class_name"].dropna().unique()),
                              key="ex_families", placeholder="All families")
    tiers = c2.multiselect("Priority tier", list(TIER_COLORS), key="ex_tiers", placeholder="All tiers")
    only_pha = c3.toggle("Official PHAs only", key="ex_only_pha")
    view = df
    if families:
        view = view[view["class_name"].isin(families)]
    if tiers:
        view = view[view["tier"].isin(tiers)]
    if only_pha:
        view = view[view["pha"]]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Objects in view", f"{len(view):,}", border=True)
    k2.metric("Official PHAs", f"{int(view['pha'].sum()):,}",
              f"{view['pha'].mean():.1%} of the view" if len(view) else None, delta_color="off", delta_arrow="off", border=True)
    k3.metric("Measured diameters", f"{(view['diameter_source'] == 'measured').sum():,}",
              "the rest is estimated from H", delta_color="off", delta_arrow="off", border=True)
    k4.metric("Approach < 20 LD by 2035", f"{(view['closest_approach_LD'] < 20).sum():,}", border=True)

    if view.empty:
        st.info("No object matches these filters.")
        return

    st.markdown("##### Raw data sample")
    st.dataframe(view[SAMPLE_COLS].head(500), hide_index=True, height=260,
                 column_config={"diameter_m": st.column_config.NumberColumn(format="%.0f"),
                                "pha": st.column_config.CheckboxColumn("pha")})

    with st.expander("Summary statistics – `df.describe()`", expanded=False):
        st.dataframe(view[[c for c in HIST_FEATURES if c != "pha_proba"]].describe().T, height=420)

    st.markdown("##### Feature distributions")
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    feature = c1.selectbox("Feature", HIST_FEATURES, index=HIST_FEATURES.index("moid_au"),
                           format_func=lambda c: FEATURE_LABELS.get(c, c), key="ex_feature")
    color_label = c2.selectbox("Colour overlay", list(COLOR_BY), key="ex_color")
    log_x = c3.toggle("log₁₀ scale", value=feature in LOG_FRIENDLY, key=f"ex_log_{feature}")
    normalize = c4.toggle("Normalize", value=True, key="ex_norm",
                          help="Each group sums to 100%: compares shapes despite the 6% PHA imbalance.")
    color_col, palette = COLOR_BY[color_label]

    data = view[[feature, color_col]].dropna()
    x_label = FEATURE_LABELS.get(feature, feature)
    if log_x:
        data = data[data[feature] > 0].assign(**{feature: lambda d: np.log10(d[feature])})
        x_label = f"log₁₀ {x_label}"
    fig = px.histogram(data, x=feature, color=color_col, color_discrete_map=palette, nbins=80,
                       barmode="overlay", opacity=0.65, histnorm="percent" if normalize else None,
                       category_orders={"tier": list(TIER_COLORS), "pha_label": ["Non-PHA", "PHA"]},
                       labels={feature: x_label, color_col: color_label})
    if feature == "moid_au":
        fig.add_vline(x=np.log10(0.05) if log_x else 0.05, line_dash="dash", line_color="#E5484D",
                      annotation_text="0.05 AU (PHA rule)")
    fig.update_layout(height=380, margin=dict(t=30, b=10), legend_title_text="")
    st.plotly_chart(fig, key="ex_hist")

    st.markdown("##### Why the target is hard to model: the PHA rule lives in MOID × diameter")
    sample = view.dropna(subset=["moid_au", "diameter_m"])
    sample = sample.sample(min(8000, len(sample)), random_state=42)
    fig = px.scatter(sample, x="moid_au", y="diameter_m", color="tier", color_discrete_map=TIER_COLORS,
                     log_x=True, log_y=True, opacity=0.55, hover_name="full_name",
                     hover_data={"pha": True, "tier": True, "moid_au": ":.4f", "diameter_m": ":.0f"},
                     category_orders={"tier": list(TIER_COLORS)},
                     labels={"moid_au": "MOID (AU, log)", "diameter_m": "Diameter (m, log)"})
    fig.update_traces(marker_size=4)
    fig.add_vline(x=0.05, line_dash="dash", line_color="#E5484D", annotation_text="MOID 0.05 AU")
    fig.add_hline(y=140, line_dash="dash", line_color="#E5484D", annotation_text="140 m")
    fig.update_layout(height=430, margin=dict(t=30, b=10), legend_title_text="")
    st.plotly_chart(fig, key="ex_scatter")
    st.caption("Official PHAs sit in the lower-left-to-upper-left box (MOID ≤ 0.05 AU, D ≥ 140 m). Because the "
               "label is a deterministic rule on MOID and H, those two columns are excluded from the "
               "classifier (see tab 2), and the policy reuses them explicitly instead (tab 4). "
               f"Showing a random sample of {len(sample):,} objects.")
