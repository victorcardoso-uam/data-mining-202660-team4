"""Tab 3 – Multi-Model Benchmark Suite."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import load_json

MODEL_COLORS = {"Decision Tree": "#3B82C4", "Random Forest": "#F08A3C", "Neural Network (MLP)": "#A78BFA",
                "Rule: crosses Earth's orbit": "#7D8597"}
LEARNED = ["Decision Tree", "Random Forest", "Neural Network (MLP)"]


def scorecard(bench: dict, mode: str) -> pd.DataFrame:
    rows = []
    for name, b in bench["classification"].items():
        m = b["operating"] if (mode == "operating" or name.startswith("Rule")) else b["default_0.5"]
        rows.append({"model": name, "task": "PHA classification", "threshold": m["threshold"],
                     "Accuracy": m["accuracy"], "Precision": m["precision"], "Recall": m["recall"],
                     "F1": m["f1"], "ROC-AUC": None if name.startswith("Rule") else m["roc_auc"],
                     "PR-AUC": None if name.startswith("Rule") else m["pr_auc"],
                     "Missed PHAs": m["missed_pha"], "% catalogue monitored": m["share_monitored"] * 100,
                     "R²": None, "Silhouette": None})
    reg = bench["regression"]["Physics-guided Random Forest"]
    rows.append({"model": "Physics-guided Random Forest", "task": "Diameter regression", "R²": reg["r2"]})
    base = bench["regression"]["Catalogue formula (albedo 0.14)"]
    rows.append({"model": "Catalogue formula (albedo 0.14)", "task": "Diameter regression", "R²": base["r2"]})
    rows.append({"model": f"K-Means (K = {bench['clustering']['k']})", "task": "Risk-profile clustering",
                 "Silhouette": bench["clustering"]["silhouette"]})
    table = pd.DataFrame(rows).set_index("model")
    num = table.columns.drop("task")
    table[num] = table[num].apply(pd.to_numeric, errors="coerce")   # None → NaN → blank cell
    return table


def render() -> None:
    st.subheader("🤖 Multi-Model Benchmark Suite")
    bench = load_json("benchmark.json")
    st.caption(f"All metrics are computed on the **held-out test set** ({bench['test_size']:,} objects, "
               f"{bench['test_pha']} PHAs) that no model saw during training or threshold selection.")

    mode = st.radio("Decision threshold", ["operating", "default"], horizontal=True, key="bm_mode",
                    format_func=lambda m: (f"Operating threshold ({bench['recall_target']:.0%} recall target)"
                                           if m == "operating" else "Default 0.5"))
    table = scorecard(bench, mode)
    pct = {c: st.column_config.NumberColumn(format="%.3f") for c in
           ["threshold", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "R²", "Silhouette"]}
    pct["% catalogue monitored"] = st.column_config.NumberColumn(format="%.1f%%")
    st.dataframe(table, column_config=pct)

    clf = table[table["task"] == "PHA classification"].loc[LEARNED]
    best_auc = clf["PR-AUC"].idxmax()
    st.info(
        f"**Reading the table.** Accuracy is misleading here (94% of objects are not PHAs: always answering "
        f"\"safe\" scores 94%). The PDCO mandate is *zero false negatives*, so we fix recall near "
        f"{bench['recall_target']:.0%} and compare **how much of the catalogue each model forces us to "
        f"monitor** and the ranking quality (PR-AUC). Best ranking: **{best_auc}** "
        f"(PR-AUC {clf.loc[best_auc, 'PR-AUC']:.2f}).", icon="💡")

    st.markdown("##### Confusion matrices (test set)")
    cols = st.columns(3)
    for col, name in zip(cols, LEARNED):
        b = bench["classification"][name]
        m = b["operating"] if mode == "operating" else b["default_0.5"]
        cm = m["confusion_matrix"]
        fig = px.imshow(cm, text_auto=",d", color_continuous_scale="Blues", aspect="auto",
                        x=["Pred. non-PHA", "Pred. PHA"], y=["Actual non-PHA", "Actual PHA"])
        fig.update_layout(height=290, margin=dict(t=50, b=10, l=10, r=10), coloraxis_showscale=False,
                          title=dict(text=f"{name}<br><sup>{m['missed_pha']} PHAs missed · "
                                          f"{m['share_monitored']:.0%} monitored</sup>", font_size=14))
        col.plotly_chart(fig, key=f"bm_cm_{name}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### ROC curves")
        fig = go.Figure()
        for name in LEARNED:
            roc = bench["classification"][name]["roc_curve"]
            auc = bench["classification"][name]["operating"]["roc_auc"]
            fig.add_scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=f"{name} ({auc:.3f})",
                            line=dict(color=MODEL_COLORS[name], width=2.5))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="random", line=dict(dash="dot", color="#7D8597"))
        fig.update_layout(height=380, margin=dict(t=20, b=10), xaxis_title="False positive rate",
                          yaxis_title="True positive rate (recall)", legend=dict(x=0.35, y=0.08))
        st.plotly_chart(fig, key="bm_roc")
    with c2:
        st.markdown("##### Neural network loss convergence")
        h = bench["nn_history"]
        fig = go.Figure()
        fig.add_scatter(x=h["epoch"], y=h["train_loss"], name="train log-loss", line=dict(color="#A78BFA", width=2.5))
        fig.add_scatter(x=h["epoch"], y=h["val_loss"], name="validation log-loss",
                        line=dict(color="#F08A3C", width=2.5))
        fig.add_scatter(x=h["epoch"], y=h["val_pr_auc"], name="validation PR-AUC", yaxis="y2",
                        line=dict(color="#7D8597", dash="dot"))
        fig.add_vline(x=h["best_epoch"], line_dash="dash", line_color="#E5484D",
                      annotation_text=f"kept: epoch {h['best_epoch']}")
        fig.update_layout(height=380, margin=dict(t=20, b=10), xaxis_title="Epoch", yaxis_title="Log-loss",
                          yaxis2=dict(title="PR-AUC", overlaying="y", side="right", showgrid=False),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, key="bm_loss")
        gap = h["val_loss"][h["best_epoch"] - 1] - h["train_loss"][h["best_epoch"] - 1]
        st.caption(f"MLP 64→32 ReLU, Adam, L2 α = 0.001, batch 256, early stopping on a 20% validation split "
                   f"of the training set (patience 12). Train/validation gap at the kept epoch: {gap:+.4f}: "
                   f"after that point the validation loss stops improving, i.e. the network starts to overfit.")

    with st.expander("Other tasks: diameter regression and risk-profile clustering"):
        c1, c2 = st.columns(2)
        reg = pd.DataFrame(bench["regression"]).T
        c1.markdown("**Diameter regression (notebook 04, test set)**")
        c1.dataframe(reg.round(3))
        k = pd.DataFrame(bench["clustering"]["k_selection"])
        fig = px.line(k, x="K", y="silhouette", markers=True)
        fig.add_vline(x=bench["clustering"]["k"], line_dash="dash", line_color="#E5484D", annotation_text="K = 4")
        fig.update_layout(height=260, margin=dict(t=30, b=10), title="K-Means silhouette by K (notebook 06)")
        c2.plotly_chart(fig, key="bm_k")
