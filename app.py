"""Planetary Defense – a simple dashboard for the NASA/JPL PDCO (Team 4, Milestone 3).

Run from the repository root:   uv run streamlit run app.py

The app only READS files that were saved by nasa-project/src/export_app_artifacts.py:
    models/*.joblib       the trained models
    models/*.json         test-set metrics (benchmark.json) and settings (metadata.json)
    data/neo_catalogue.csv.gz   the cleaned catalogue
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

MODELS = Path(__file__).parent / "models"
DATA = Path(__file__).parent / "data"
PHA_COLORS = {"PHA": "#F08A3C", "Non-PHA": "#3B82C4"}

st.set_page_config(page_title="Planetary Defense", page_icon="☄️", layout="wide")


# ---------------------------------------------------------------------------
# Loading (cached: the files are read once, not at every click)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA / "neo_catalogue.csv.gz", low_memory=False)
    df["pha"] = df["pha"].astype(bool)
    df["pha_label"] = np.where(df["pha"].astype(bool), "PHA", "Non-PHA")
    df["diameter_m"] = df["diameter_km"] * 1000
    return df


@st.cache_data
def load_json(name: str) -> dict:
    return json.loads((MODELS / name).read_text())


@st.cache_resource
def load_model(file: str):
    return joblib.load(MODELS / file)


df = load_data()
meta = load_json("metadata.json")
bench = load_json("benchmark.json")
FEATURES = meta["features"]["numeric"] + meta["features"]["categorical"]


@st.cache_data
def scaled_features() -> pd.DataFrame:
    """The catalogue after the preprocessing saved inside the Random Forest pipeline (scaling + one-hot)."""
    prep = load_model(meta["models"]["Random Forest"]["file"]).named_steps["prep"]
    data = load_data()
    return pd.DataFrame(prep.transform(data[FEATURES]), columns=prep.get_feature_names_out(), index=data.index)

# ---------------------------------------------------------------------------
# Header and KPI cards
# ---------------------------------------------------------------------------
st.title("☄️ Planetary Defense: which asteroids should we watch?")
st.caption("Team 4 · Data Mining · Client: NASA/JPL Planetary Defense Coordination Office")

rf = bench["classification"]["Random Forest"]["operating"]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Near-Earth asteroids", f"{len(df):,}", border=True)
k2.metric(f"Official PHAs ({df['pha'].mean():.0%})", f"{df['pha'].sum():,}", border=True)
k3.metric("PHAs caught by our model (recall)", f"{rf['recall']:.1%}", border=True)
k4.metric("Share of catalogue to monitor", f"{rf['share_monitored']:.0%}", border=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Explorer", "⚙️ Preprocessing", "🤖 Model Benchmark", "🔮 What-If"])

# ---------------------------------------------------------------------------
# Tab 1 – Data Explorer
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("What does the data look like?")
    st.write("Each row is one asteroid. The target is `pha` (Potentially Hazardous Asteroid): "
             "an asteroid whose orbit passes within 0.05 AU of Earth's orbit (MOID) and that is at least ~140 m wide.")
    st.dataframe(df[["full_name", "pha", "a", "e", "i", "moid_au", "diameter_m", "condition_code"]].head(100),
                 hide_index=True, height=250)

    st.markdown("**Summary statistics** (`df.describe()`)")
    stats_cols = ["a", "e", "i", "moid_au", "diameter_m", "H", "condition_code", "data_arc_years", "first_obs_year"]
    st.dataframe(df[stats_cols].describe().T.round(2))
    st.caption("For each variable: how many values (count), the average (mean), the spread (std), the range (min/max) "
               "and the quartiles. For example, the median asteroid is about 60 m wide, but the largest is about 38 km: "
               "most near-Earth asteroids are small.")

    feature = st.selectbox("Pick a variable to see its distribution",
                           ["moid_au", "diameter_m", "a", "e", "i", "condition_code", "data_arc_years"])
    fig = px.histogram(df, x=feature, color="pha_label", color_discrete_map=PHA_COLORS,
                       nbins=60, barmode="overlay", histnorm="percent")
    st.plotly_chart(fig)
    st.caption("Bars are in % of each group, so PHAs (only 6% of the data) can be compared with non-PHAs.")

    st.markdown("**The PHA rule in one picture:** PHAs are the asteroids that are both close (left) and big (top).")
    sample = df.dropna(subset=["moid_au", "diameter_m"]).sample(5000, random_state=42)
    fig = px.scatter(sample, x="moid_au", y="diameter_m", color="pha_label", color_discrete_map=PHA_COLORS,
                     log_x=True, log_y=True, opacity=0.5, labels={"moid_au": "MOID (AU)", "diameter_m": "Diameter (m)"})
    fig.add_vline(x=0.05, line_dash="dash")
    fig.add_hline(y=140, line_dash="dash")
    st.plotly_chart(fig)

# ---------------------------------------------------------------------------
# Tab 2 – Preprocessing & correlation
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("How did we prepare the data?")
    st.markdown(f"""
1. **Removed columns that give away the answer.** `pha` is *defined* from MOID and size (H, diameter).
   If the model sees them it just copies the rule. So the model only uses: {", ".join(f"`{c}`" for c in FEATURES)}.
2. **Filled missing values** with the median of the training set.
3. **Scaled numbers** (z-score) and **one-hot encoded** the orbital family `class`.
4. **Split** the data into train / test, so all scores below are on asteroids the models never saw.
""")

    missing = pd.Series(meta["missing_share_raw"]).mul(100).sort_values(ascending=False)
    fig = px.bar(missing, labels={"index": "column", "value": "% missing"}, title="Missing values per column")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)
    st.caption("`albedo` and `rot_per` are almost empty, so we dropped them instead of filling them.")

    # ---- Feature scaling: before / after the StandardScaler saved in the .joblib model
    st.markdown("**Feature scaling** (StandardScaler, loaded from the Random Forest's .joblib file)")
    scaled = scaled_features()
    feat = st.selectbox("Pick a variable", meta["features"]["numeric"], key="pp_feature")
    keep = df[feat] <= df[feat].quantile(0.99)          # hide the 1% extreme values so the shape is visible
    c1, c2 = st.columns(2)
    fig = px.histogram(df[keep], x=feat, nbins=60, title=f"Before: raw {feat}")
    c1.plotly_chart(fig, key="pp_raw")
    fig = px.histogram(scaled[keep], x=feat, nbins=60, title=f"After: scaled {feat}", color_discrete_sequence=["#F08A3C"])
    c2.plotly_chart(fig, key="pp_scaled")
    st.caption(f"After scaling, {feat} has mean {round(scaled[feat].mean(), 2) + 0.0:.2f} and standard deviation {scaled[feat].std():.2f}: "
               "every variable ends up on the same scale, so none dominates just because of its units. "
               "Very skewed variables (a, ad, years observed) get a log first.")

    # ---- One-hot encoding of the orbital family
    st.markdown("**One-hot encoding** of the orbital family `class`")
    examples = df.groupby("class").head(1)
    onehot_cols = [c for c in scaled.columns if c.startswith("class_")]
    st.dataframe(pd.concat([examples[["full_name", "class"]], scaled.loc[examples.index, onehot_cols].astype(int)], axis=1),
                 hide_index=True)
    st.caption("A model cannot read text, so each family becomes its own 0/1 column. "
               "APO = Apollo, AMO = Amor, ATE = Aten, IEO = Atira.")

    # ---- Interactive Pearson correlation heatmap
    st.markdown("**Pearson correlation matrix** (1 = move together, −1 = move in opposite directions)")
    corr_cols = ["a", "e", "i", "q", "ad", "condition_code", "data_arc_years", "first_obs_year", "H", "pha"]
    picked = st.multiselect("Variables to include", corr_cols, default=corr_cols, key="pp_corr_cols")
    if len(picked) >= 2:
        corr = df[picked].astype(float).corr(method="pearson")
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(height=550)
        st.plotly_chart(fig, key="pp_corr")
    else:
        st.info("Pick at least two variables.")

# ---------------------------------------------------------------------------
# Tab 3 – Model benchmark
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Which model is best?")
    st.write(f"Three classifiers trained on the same data, tested on {bench['test_size']:,} unseen asteroids "
             f"({bench['test_pha']} PHAs). Each model's threshold is tuned to catch about 99% of PHAs.")

    models = ["Decision Tree", "Random Forest", "Neural Network (MLP)"]
    rows = []
    for name in models:
        m = bench["classification"][name]["operating"]
        rows.append({"Model": name, "Accuracy": m["accuracy"], "Precision": m["precision"], "Recall": m["recall"],
                     "F1": m["f1"], "ROC-AUC": m["roc_auc"], "PHAs missed": m["missed_pha"],
                     "% catalogue to monitor": m["share_monitored"] * 100})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
    st.info("**How to read it:** all models catch ~99% of PHAs (recall). The winner is the one that needs to "
            "monitor the **smallest share of the catalogue** to do so: the **Random Forest**.")

    st.markdown("**Confusion matrices**")
    for col, name in zip(st.columns(3), models):
        cm = bench["classification"][name]["operating"]["confusion_matrix"]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title=name,
                        x=["Pred. safe", "Pred. PHA"], y=["Real safe", "Real PHA"])
        fig.update_layout(coloraxis_showscale=False, height=300)
        col.plotly_chart(fig)

    st.markdown("**Neural network training:** the loss goes down, then the validation loss stops improving "
                "(the network starts to memorize), so we keep the best epoch.")
    h = bench["nn_history"]
    loss = pd.DataFrame({"epoch": h["epoch"], "train": h["train_loss"], "validation": h["val_loss"]})
    fig = px.line(loss, x="epoch", y=["train", "validation"], labels={"value": "loss", "variable": ""})
    fig.add_vline(x=h["best_epoch"], line_dash="dash", annotation_text="best epoch")
    st.plotly_chart(fig)

    c1, c2 = st.columns(2)
    c1.metric("Diameter regression (Random Forest) – R²",
              f"{bench['regression']['Physics-guided Random Forest']['r2']:.2f}", border=True)
    c2.metric(f"Clustering (K-Means, K={bench['clustering']['k']}) – Silhouette",
              f"{bench['clustering']['silhouette']:.2f}", border=True)

# ---------------------------------------------------------------------------
# Tab 4 – What-If
# ---------------------------------------------------------------------------
DEFAULTS = {"a": 1.8, "e": 0.35, "i": 10.0, "cc": 0, "arc": 20.0, "year": 2000,
            "diam": 100, "moid": 0.03, "vinf": 10.0, "ld": 50}


def reset():
    """Put every slider back to its default value (stored in st.session_state)."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


with tab4:
    st.subheader("Try your own asteroid")
    st.button("↺ Reset sliders", on_click=reset)

    left, right = st.columns(2)

    # ---- Part A: the classifier --------------------------------------------
    with left:
        st.markdown("#### A · Is it a PHA? (machine learning)")
        model_name = st.selectbox("Model (which trained model makes the prediction)", list(meta["models"]), key="model",
                                  help="Each model was trained on the same data and saved as a .joblib file.")
        a = st.slider("Semi-major axis a (size of the orbit, AU)", 0.6, 4.0, DEFAULTS["a"], 0.01, key="a",
                      help="Average distance from the Sun. 1 AU = Earth's distance from the Sun (150 million km).")
        e = st.slider("Eccentricity e (how stretched the orbit is)", 0.0, 0.95, DEFAULTS["e"], 0.01, key="e",
                      help="0 = circle, close to 1 = very elongated. A more stretched orbit comes closer to the Sun, "
                           "and can cross Earth's orbit.")
        i = st.slider("Inclination i (tilt of the orbit, °)", 0.0, 60.0, DEFAULTS["i"], 0.5, key="i",
                      help="Angle between the asteroid's orbit and Earth's orbit. 0° = same plane as Earth.")
        cc = st.slider("Orbit uncertainty (how well we know its path: 0 = precise, 9 = poor)", 0, 9, DEFAULTS["cc"], key="cc",
                       help="NASA's condition code. A poorly known orbit (7 or more) needs urgent observation.")
        arc = st.slider("Years observed (how long it has been tracked)", 0.0, 50.0, DEFAULTS["arc"], 0.5, key="arc",
                        help="Time between the first and last observation. Long-tracked asteroids tend to be bigger.")
        year = st.slider("Discovery year (when it was first seen)", 1950, 2025, DEFAULTS["year"], key="year",
                         help="Big, bright asteroids were found first; recent discoveries are mostly small.")

        q, ad = a * (1 - e), a * (1 + e)   # closest and farthest distance from the Sun
        family = ("IEO" if ad < 0.983 else "ATE") if a < 1 else ("APO" if q < 1.017 else "AMO")
        X = pd.DataFrame([{"a": a, "e": e, "i": i, "q": q, "ad": ad, "condition_code": cc,
                           "data_arc_years": arc, "first_obs_year": year, "class": family}])[FEATURES]

        info = meta["models"][model_name]
        proba = load_model(info["file"]).predict_proba(X)[0, 1]
        flagged = proba >= info["threshold"]
        st.metric("PHA probability", f"{proba:.1%}", border=True)
        if flagged:
            st.warning(f"⚠️ Flagged as a possible PHA (threshold {info['threshold']:.3f})")
        else:
            st.success(f"✅ Not flagged (threshold {info['threshold']:.3f})")

    # ---- Part B: size and speed -> impact energy and priority ---------------
    with right:
        st.markdown("#### B · How bad would an impact be? (physics)")
        diam = st.slider("Diameter (width of the asteroid, m)", 10, 1000, DEFAULTS["diam"], 10, key="diam",
                         help="140 m is the size limit of the official PHA rule. Impact energy grows with the cube of the size.")
        moid = st.slider("MOID (closest distance between its orbit and Earth's orbit, AU)", 0.0, 0.3, DEFAULTS["moid"],
                         0.005, key="moid",
                         help="0.05 AU (about 7.5 million km, 19.5 Moon distances) is the distance limit of the PHA rule.")
        vinf = st.slider("Velocity (how fast it approaches Earth, km/s)", 1.0, 40.0, DEFAULTS["vinf"], 0.5, key="vinf",
                         help="Speed before Earth's gravity pulls it in. Earth adds 11.2 km/s at impact.")
        ld = st.slider("Closest approach before 2035 (in Earth–Moon distances)", 0, 100, DEFAULTS["ld"], key="ld",
                       help="How close it will actually pass Earth. 1 = 384,000 km. Below 20 counts as urgent.")

        p = meta["policy"]
        mass = p["density_kg_m3"] * (4 / 3) * np.pi * (diam / 2) ** 3            # kg
        v = np.sqrt(vinf ** 2 + p["escape_velocity_kms"] ** 2) * 1000              # m/s at impact
        energy_mt = 0.5 * mass * v ** 2 / p["mt_tnt_j"]                            # megatons of TNT
        st.metric("Impact energy", f"{energy_mt:,.2f} Mt TNT",
                  f"≈ {energy_mt / 0.015:,.0f} × Hiroshima", delta_color="off", border=True)

        is_pha_rule = moid <= 0.05 and diam >= 140
        urgent = ld < 20 or cc >= 7                     # passes very close soon, or orbit badly known
        if is_pha_rule and urgent:
            st.error("🔴 Tier 1 – track now: meets the PHA rule AND is urgent (close pass or uncertain orbit).")
        elif is_pha_rule or flagged:
            st.warning("🟠 Tier 2 – monitor yearly: meets the PHA rule or was flagged by the model.")
        else:
            st.success("🟢 Tier 3 – routine: too far or too small to be a PHA.")
        st.caption("PHA rule: MOID ≤ 0.05 AU **and** diameter ≥ 140 m. Urgent: approach < 20 Moon distances "
                   "**or** orbit uncertainty ≥ 7. (Simplified version of the notebook 07 policy.)")

    # ---- Priority list: the policy applied to the real catalogue -----------
    st.divider()
    st.subheader("📋 Priority list: real asteroids to observe")
    st.write("Our 3-tier policy (notebook 07) applied to the whole catalogue: every asteroid gets a tier and a reason, "
             "most urgent first. This is the observation plan we hand to the client.")
    counts = df["tier"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Tier 1 · observe this cycle", f"{counts.get('Tier 1', 0):,}", border=True)
    c2.metric("🟠 Tier 2 · next cycle", f"{counts.get('Tier 2', 0):,}", border=True)
    c3.metric("🟢 Tier 3 · passive surveillance", f"{counts.get('Tier 3', 0):,}", border=True)

    c1, c2 = st.columns([1, 2])
    tier_pick = c1.selectbox("Show tier", ["Tier 1", "Tier 2", "Tier 3"], key="pl_tier")
    search = c2.text_input("Or search an asteroid by name (all tiers)", key="pl_search", placeholder="e.g. 2002 TB70")
    if search:
        view = df[df["full_name"].str.contains(search.strip(), case=False, na=False, regex=False)]
    else:
        view = df[df["tier"] == tier_pick]
    table = view.sort_values("rank").assign(
        pha_proba=lambda d: d["pha_proba"] * 100,
        closest_approach_date=lambda d: d["closest_approach_date"].astype(str).str[:10].replace("nan", ""))
    st.dataframe(
        table[["rank", "full_name", "tier", "pha", "pha_proba", "diameter_m", "moid_au", "closest_approach_date",
               "closest_approach_LD", "impact_energy_Mt_high", "reasons"]].head(500),
        hide_index=True, height=360,
        column_config={
            "rank": st.column_config.NumberColumn("Rank"),
            "full_name": "Asteroid",
            "tier": "Tier",
            "pha": st.column_config.CheckboxColumn("Official PHA"),
            "pha_proba": st.column_config.NumberColumn("Model probability", format="%.0f%%"),
            "diameter_m": st.column_config.NumberColumn("Diameter (m)", format="%.0f"),
            "moid_au": st.column_config.NumberColumn("MOID (AU)", format="%.3f"),
            "closest_approach_date": "Closest approach",
            "closest_approach_LD": st.column_config.NumberColumn("Distance (Moon distances)", format="%.1f"),
            "impact_energy_Mt_high": st.column_config.NumberColumn("Impact energy, pessimistic (Mt TNT)", format="%.0f"),
            "reasons": "Why this tier",
        })
    st.caption(f"{len(view):,} asteroids match (first 500 shown). Diameter is our regression estimate unless measured; "
               "impact energy uses the pessimistic (upper-bound) diameter.")
