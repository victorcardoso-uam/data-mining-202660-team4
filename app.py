"""Planetary Defense – NEO prioritization dashboard for the NASA/JPL PDCO.

Run from the repository root:

    uv run streamlit run app.py        (or: pip install -r requirements.txt && streamlit run app.py)

The app only reads versioned artifacts: models/*.joblib, models/*.json, data/neo_catalogue.csv.gz and
nasa-project/reports/. To rebuild them, see nasa-project/src/export_app_artifacts.py.
"""

import streamlit as st

from dashboard import tab_benchmark, tab_explorer, tab_preprocessing, tab_whatif
from dashboard.common import MODELS_DIR, load_catalogue, load_json

st.set_page_config(page_title="Planetary Defense · NEO Prioritization", page_icon="☄️", layout="wide")

missing = [f for f in ["metadata.json", "benchmark.json"] if not (MODELS_DIR / f).exists()]
if missing:
    st.error(f"Missing artifacts in models/: {missing}. Run `uv run python nasa-project/src/export_app_artifacts.py`.")
    st.stop()

df = load_catalogue()
bench = load_json("benchmark.json")
chosen = bench["classification"]["Random Forest"]["operating"]

st.title("☄️ Planetary Defense · Near-Earth Object Prioritization")
st.caption("Client: NASA/JPL Planetary Defense Coordination Office · Team 4 · Data Mining & Modern AI Systems "
           "(IIND4417, 202660)")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Catalogued NEOs", f"{len(df):,}", border=True)
k2.metric("Official PHAs", f"{int(df['pha'].sum()):,}", f"{df['pha'].mean():.1%} of the catalogue",
          delta_color="off", delta_arrow="off", border=True)
k3.metric("Tier 1 · track this cycle", f"{(df['tier'] == 'Tier 1').sum():,}",
          f"{(df['tier'] == 'Tier 1').mean():.1%} of the catalogue", delta_color="off", delta_arrow="off", border=True)
k4.metric("PHA recall (test, RF)", f"{chosen['recall']:.1%}", f"{chosen['missed_pha']} missed", delta_color="off", delta_arrow="off",
          border=True)
rule = bench["classification"]["Rule: crosses Earth's orbit"]["operating"]
k5.metric("Catalogue to monitor", f"{chosen['share_monitored']:.0%}",
          f"vs {rule['share_monitored']:.0%} with the orbit-crossing rule",
          delta_color="off", delta_arrow="off", border=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Explorer & EDA", "⚙️ Preprocessing & Correlation",
                                  "🤖 Multi-Model Benchmark", "🔮 What-If Inference"])
with tab1:
    tab_explorer.render(df)
with tab2:
    tab_preprocessing.render(df)
with tab3:
    tab_benchmark.render()
with tab4:
    tab_whatif.render(df)
