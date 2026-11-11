"""Tab 4 – Real-Time What-If Scenario Inference Engine."""

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.common import (CLASS_NAMES, COST_PER_HOUR_USD, OBS_PER_YEAR, TIER_COLORS, assess_risk,
                              fmt_energy, load_json, load_model, load_report, orbit_class)

CUSTOM = "__custom__"
# Slider ranges: (min, max, step)
RANGES = {
    "wi_a": (0.6, 4.0, 0.01), "wi_e": (0.0, 0.95, 0.01), "wi_i": (0.0, 75.0, 0.5),
    "wi_cc": (0, 9, 1), "wi_arc": (0.0, 80.0, 0.1), "wi_year": (1900, 2026, 1),
    "wi_H": (14.0, 30.0, 0.1), "wi_diam": (1, 3000, 1), "wi_vinf": (1.0, 40.0, 0.5),
    "wi_moid": (0.0, 0.30, 0.001), "wi_ld": (0.5, 100.0, 0.5),
}
DEFAULTS = {"wi_object": CUSTOM, "wi_model": "Random Forest", "wi_a": 1.40, "wi_e": 0.42, "wi_i": 8.0,
            "wi_cc": 3, "wi_arc": 6.0, "wi_year": 2019, "wi_mode": "Estimate from H (regression model)",
            "wi_H": 24.5, "wi_diam": 250, "wi_vinf": 12.0, "wi_moid": 0.030, "wi_has_app": True,
            "wi_ld": 15.0, "wi_recent": False}
REFERENCE_EVENTS = [(0.015, "Hiroshima bomb"), (0.5, "Chelyabinsk 2013"), (12, "Tunguska 1908"),
                    (1e8, "Chicxulub (dinosaurs)")]


def _clip(key, value):
    lo, hi, step = RANGES[key]
    value = float(np.clip(value, lo, hi))
    if isinstance(lo, int):
        return int(round(value))
    return round(lo + round((value - lo) / step) * step, 3)   # snap to the slider grid


def _init_state() -> None:
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)


def _load_object(df: pd.DataFrame, extraction: pd.Timestamp) -> None:
    """Callback: copy a catalogue object into the scenario sliders (st.session_state)."""
    pdes = st.session_state["wi_object"]
    if pdes == CUSTOM:
        return
    r = df.loc[df["pdes"] == pdes].iloc[0]
    st.session_state.update({
        "wi_a": _clip("wi_a", r["a"]), "wi_e": _clip("wi_e", r["e"]), "wi_i": _clip("wi_i", r["i"]),
        "wi_cc": _clip("wi_cc", r["condition_code"] if pd.notna(r["condition_code"]) else 5),
        "wi_arc": _clip("wi_arc", r["data_arc_years"] if pd.notna(r["data_arc_years"]) else 0),
        "wi_year": _clip("wi_year", r["first_obs_year"] if pd.notna(r["first_obs_year"]) else 2026),
        "wi_H": _clip("wi_H", r["H"]),
        "wi_mode": "Estimate from H (regression model)" if r["diameter_source"] == "predicted"
        else "Set the diameter directly (measured)",
        "wi_diam": _clip("wi_diam", r["diameter_km"] * 1000),
        "wi_vinf": _clip("wi_vinf", r["v_inf_kms"] if pd.notna(r["v_inf_kms"]) else 12),
        "wi_moid": _clip("wi_moid", r["moid_au"]),
        "wi_has_app": bool(pd.notna(r["closest_approach_LD"])),
        "wi_ld": _clip("wi_ld", r["closest_approach_LD"] if pd.notna(r["closest_approach_LD"]) else 50),
        "wi_recent": bool(pd.notna(r["first_obs"]) and r["first_obs"] >= extraction - pd.Timedelta(days=365)),
    })


def _reset() -> None:
    st.session_state.update(DEFAULTS)


def _slider(col, label, key, fmt=None, help=None):
    lo, hi, step = RANGES[key]
    return col.slider(label, lo, hi, step=step, key=key, format=fmt, help=help)


def render(df: pd.DataFrame) -> None:
    _init_state()
    meta = load_json("metadata.json")
    policy = meta["policy"]
    extraction = pd.Timestamp(meta["extraction_date"])

    st.subheader("🔮 Real-Time What-If Scenario Inference Engine")
    st.caption("Every slider re-runs the serialized models (.joblib) live. Start from a real object of the "
               "priority queue or build a hypothetical asteroid.")

    queue = df.nsmallest(300, "rank")
    labels = {CUSTOM: "✏️ Custom object"} | {
        p: f"#{rk} · {n} · {t}" for p, rk, n, t in zip(queue["pdes"], queue["rank"], queue["full_name"], queue["tier"])}
    c1, c2, c3 = st.columns([3, 2, 1])
    c1.selectbox("Start from", list(labels), format_func=labels.get, key="wi_object",
                 on_change=_load_object, args=(df, extraction),
                 help="The 300 top-ranked objects of the notebook 07 priority queue.")
    c2.selectbox("Hazard model", list(meta["models"]), key="wi_model")
    c3.button("Reset", on_click=_reset, width="stretch")

    left, right = st.columns(2, gap="large")

    # ---- A. Orbit + observation → PHA probability -------------------------------------------
    with left:
        st.markdown("##### A · Orbit & observation history → hazard model")
        a1, a2 = st.columns(2)
        a = _slider(a1, "Semi-major axis a (AU)", "wi_a")
        e = _slider(a2, "Eccentricity e", "wi_e")
        i = _slider(a1, "Inclination i (°)", "wi_i")
        cc = _slider(a2, "Condition code (orbit uncertainty)", "wi_cc", help="0 = precise orbit, 9 = very poor")
        arc = _slider(a1, "Observation arc (years)", "wi_arc")
        year = _slider(a2, "First observation year", "wi_year")
        q, ad = a * (1 - e), a * (1 + e)
        cls = orbit_class(a, q, ad)
        st.caption(f"Derived: perihelion q = **{q:.3f} AU**, aphelion ad = **{ad:.3f} AU**, family "
                   f"**{CLASS_NAMES[cls]}**" + (" ⚠️ q ≥ 1.3 AU: not a near-Earth orbit" if q >= 1.3 else ""))

        x = pd.DataFrame([{"a": a, "e": e, "i": i, "q": q, "ad": ad, "condition_code": cc,
                           "data_arc_years": arc, "first_obs_year": year, "class": cls}])
        probas = {}
        for name, m in meta["models"].items():
            probas[name] = float(load_model(m["file"]).predict_proba(x)[0, 1])
        name = st.session_state["wi_model"]
        proba, thr = probas[name], meta["models"][name]["threshold"]
        flagged = proba >= thr

        st.progress(min(proba, 1.0), text=f"{name}: P(PHA) = **{proba:.1%}** · alert threshold {thr:.1%}")
        if flagged:
            st.warning(f"**Flagged for surveillance** by the {name}: the score is above the threshold that "
                       f"catches {meta['recall_target']:.0%} of PHAs.", icon="🛰️")
        else:
            st.success(f"Below the {name} alert threshold: orbit and tracking history look like a typical "
                       f"non-hazardous object.", icon="✅")
        st.dataframe(pd.DataFrame({"P(PHA)": probas, "threshold": {n: m["threshold"] for n, m in meta["models"].items()}})
                     .assign(flagged=lambda d: d["P(PHA)"] >= d["threshold"]),
                     column_config={"P(PHA)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                                    "threshold": st.column_config.NumberColumn(format="%.3f")})

    # ---- B. Size + encounter → policy tier -------------------------------------------------
    with right:
        st.markdown("##### B · Size & encounter geometry → tracking policy")
        mode = st.radio("Diameter", ["Estimate from H (regression model)", "Set the diameter directly (measured)"],
                        key="wi_mode", horizontal=True)
        b1, b2 = st.columns(2)
        if mode.startswith("Estimate"):
            H = _slider(b1, "Absolute magnitude H (brighter = bigger)", "wi_H")
            reg = load_model("diameter_rf.joblib")
            term = reg.predict(x.assign(H=H)[["H", "a", "e", "i", "q", "ad", "class"]])[0]
            log_d = term - H / 5
            lo_dex, hi_dex = meta["diameter_interval_dex"]
            diam_km, diam_low, diam_high = 10 ** log_d, 10 ** (log_d + lo_dex), 10 ** (log_d + hi_dex)
            albedo = 10 ** (2 * (np.log10(1329) - term))
            b2.metric("Estimated diameter", f"{diam_km * 1000:,.0f} m",
                      f"90% interval {diam_low * 1000:,.0f}–{diam_high * 1000:,.0f} m", delta_color="off", delta_arrow="off", border=True)
            st.caption(f"Physics-guided Random Forest: implied albedo **{albedo:.2f}** (catalogue assumes 0.14)."
                       + (f" ⚠️ H > {meta['diameter_h_limit']:.1f}: outside the training domain." if H > meta["diameter_h_limit"] else ""))
        else:
            diam_km = _slider(b1, "Diameter (m)", "wi_diam") / 1000
            diam_high = diam_km
            b2.metric("Diameter", f"{diam_km * 1000:,.0f} m", "measured: no uncertainty band", delta_color="off", delta_arrow="off", border=True)
        v_inf = _slider(b1, "Arrival velocity v∞ (km/s)", "wi_vinf")
        moid = _slider(b2, "MOID: minimum orbit distance (AU)", "wi_moid", fmt="%.3f")
        has_app = b1.toggle("Close approach predicted by 2035", key="wi_has_app")
        ld = _slider(b2, "Closest approach (lunar distances)", "wi_ld") if has_app else None
        recent = b1.toggle(f"Discovered < 1 year before {extraction:%Y-%m-%d}", key="wi_recent")

    risk = assess_risk(diam_km, diam_high, moid, ld, cc, recent, flagged, v_inf, policy)

    # ---- Decision panel ------------------------------------------------------------------
    st.divider()
    tier = risk["tier"]
    hours = OBS_PER_YEAR[tier]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Priority tier", tier, border=True)
    k2.metric("Pessimistic impact energy", fmt_energy(risk["energy_mt_high"]),
              f"impact speed {risk['v_impact_kms']:.1f} km/s", delta_color="off", delta_arrow="off", border=True)
    k3.metric("Telescope time", f"{hours} h / year", f"≈ ${hours * COST_PER_HOUR_USD:,} / year (assumption)",
              delta_color="off", delta_arrow="off", border=True)
    passed = [label for e_mt, label in REFERENCE_EVENTS if risk["energy_mt_high"] >= e_mt]
    k4.metric("Worse than", passed[-1] if passed else "—", "pessimistic energy vs historical events",
              delta_color="off", delta_arrow="off", border=True)

    reasons = " · ".join(risk["reasons"])
    if tier == "Tier 1":
        st.error(f"🚨 **PRIORITY TELESCOPE TRACKING – schedule this cycle.** {reasons}. Allocate "
                 f"{OBS_PER_YEAR['Tier 1']} dedicated observations per year and notify the PDCO duty officer.")
    elif tier == "Tier 2":
        st.warning(f"🔭 **Follow-up next cycle.** {reasons}. One dedicated observation per year to refine the "
                   f"orbit and the size estimate.")
    else:
        st.success(f"✅ **Passive surveillance only.** {reasons}. Automated surveys are enough; no dedicated "
                   f"telescope hours.")

    if 0 < moid <= policy["moid_max_au"] * 4:
        km = load_model("kmeans_risk_profiles.joblib")
        feats = pd.DataFrame([{"log10_diameter": np.log10(diam_km), "log10_moid": np.log10(moid + 1e-4),
                               "inclination": i, "eccentricity": e}])
        cluster = meta["kmeans_relabel"][str(int(km.predict(feats)[0]))]
        camp = load_report("06_clustering", "campaigns_by_cluster.csv", index_col=0).loc[cluster]
        st.info(f"**Risk profile (K-Means): cluster {cluster} – {camp['profile']}.** "
                f"{camp['follow-up strategy']}", icon="🧭")

    with st.expander("How this decision is made"):
        st.markdown(
            "1. **Hazard model (left):** predicts the official PHA flag from the orbit and the tracking "
            "history only (no MOID, no H). It acts as a safety net for fresh discoveries whose size and MOID "
            "are still uncertain.\n"
            "2. **Policy (right):** applies the notebook 07 rules. *Probable PHA* = MOID ≤ 0.05 AU and diameter "
            "≥ 140 m; *possible PHA* = only the pessimistic diameter reaches 140 m. Tier 1 = probable PHA **and** "
            "urgent (approach < 20 LD by 2035, condition code ≥ 7, or discovered within a year).\n"
            "3. **Energy:** E = ½ · m · v², with m from the diameter (density 2,600 kg/m³) and "
            "v = √(v∞² + 11.2²) km/s. The pessimistic diameter (upper bound of the 90% interval) is used, "
            "because the diameter is cubed.")
