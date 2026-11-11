"""Shared loaders (cached), constants and domain rules used by every tab."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "nasa-project" / "reports"

TIER_COLORS = {"Tier 1": "#E5484D", "Tier 2": "#F5A524", "Tier 3": "#7D8597"}
PHA_COLORS = {"PHA": "#F08A3C", "Non-PHA": "#3B82C4"}
CLASS_NAMES = {"APO": "Apollo", "AMO": "Amor", "ATE": "Aten", "IEO": "Atira"}
OBS_PER_YEAR = {"Tier 1": 4, "Tier 2": 1, "Tier 3": 0}   # notebook 07 (illustrative)
COST_PER_HOUR_USD = 5_000

FEATURE_LABELS = {
    "a": "a – semi-major axis (AU)",
    "e": "e – eccentricity",
    "i": "i – inclination (°)",
    "q": "q – perihelion distance (AU)",
    "ad": "ad – aphelion distance (AU)",
    "condition_code": "condition code (0 = precise orbit, 9 = poor)",
    "data_arc_years": "observation arc (years)",
    "first_obs_year": "first observation year",
    "H": "H – absolute magnitude",
    "moid_au": "MOID (AU)",
    "diameter_m": "diameter (m)",
    "v_inf_kms": "arrival velocity v∞ (km/s)",
    "closest_approach_LD": "closest approach by 2035 (lunar distances)",
    "impact_energy_Mt_high": "pessimistic impact energy (Mt TNT)",
    "pha_proba": "PHA probability (Random Forest)",
}


@st.cache_data(show_spinner="Loading the NEO catalogue…")
def load_catalogue() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "neo_catalogue.csv.gz", dtype={"pdes": str}, low_memory=False,
                     parse_dates=["closest_approach_date", "first_obs"])
    df["pha"] = df["pha"].astype(bool)
    df["pha_label"] = np.where(df["pha"], "PHA", "Non-PHA")
    df["diameter_m"] = df["diameter_km"] * 1000
    df["class_name"] = df["class"].map(CLASS_NAMES).fillna(df["class"])
    df["full_name"] = df["full_name"].str.strip()
    return df


@st.cache_data
def load_json(name: str) -> dict:
    return json.loads((MODELS_DIR / name).read_text(encoding="utf-8"))


@st.cache_data
def load_report(step: str, name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(REPORTS_DIR / step / name, **kwargs)


@st.cache_resource(show_spinner="Loading serialized models (.joblib)…")
def load_model(file: str):
    return joblib.load(MODELS_DIR / file)


def orbit_class(a: float, q: float, ad: float) -> str:
    """JPL near-Earth object families from the orbital elements."""
    if a < 1.0:
        return "IEO" if ad < 0.983 else "ATE"
    return "APO" if q < 1.017 else "AMO"


def assess_risk(diameter_km: float, diameter_high_km: float, moid_au: float, approach_ld: float | None,
                condition_code: int, recent_discovery: bool, model_flag: bool, v_inf_kms: float,
                policy: dict) -> dict:
    """Three-tier prioritization policy of notebook 07, applied to a single object."""
    v_impact = np.sqrt(v_inf_kms ** 2 + policy["escape_velocity_kms"] ** 2) * 1e3

    def energy_mt(d_km):
        mass = policy["density_kg_m3"] * np.pi / 6 * (d_km * 1e3) ** 3
        return 0.5 * mass * v_impact ** 2 / policy["mt_tnt_j"]

    close_orbit = moid_au <= policy["moid_max_au"]
    probable = close_orbit and diameter_km >= policy["min_diameter_km"]
    possible = close_orbit and not probable and diameter_high_km >= policy["min_diameter_km"]
    close_approach = approach_ld is not None and approach_ld < policy["urgent_approach_ld"]
    uncertain = condition_code >= policy["uncertain_condition_code"]
    urgent = close_approach or uncertain or recent_discovery
    safety_net = recent_discovery and model_flag and diameter_high_km >= policy["min_diameter_km"]

    if probable and urgent:
        tier = "Tier 1"
    elif probable or possible or safety_net:
        tier = "Tier 2"
    else:
        tier = "Tier 3"

    reasons = []
    if probable:
        reasons.append("probable PHA (MOID ≤ 0.05 AU and diameter ≥ 140 m)")
    if possible:
        reasons.append("possible PHA (only the pessimistic diameter reaches 140 m)")
    if close_approach:
        reasons.append(f"approach at {approach_ld:.1f} LD before 2035")
    if uncertain:
        reasons.append(f"uncertain orbit (condition code {condition_code})")
    if recent_discovery:
        reasons.append("recent discovery")
    if safety_net and not (probable or possible):
        reasons.append("flagged by the PHA model (safety net)")
    if not close_orbit:
        reasons.append(f"MOID {moid_au:.3f} AU > 0.05 AU: the orbit never comes close enough to be a PHA")
    elif not (probable or possible):
        reasons.append("too small: even the pessimistic diameter stays under 140 m")

    return {
        "tier": tier,
        "probable": probable,
        "possible": possible,
        "urgent": urgent,
        "reasons": reasons,
        "energy_mt": energy_mt(diameter_km),
        "energy_mt_high": energy_mt(diameter_high_km),
        "v_impact_kms": v_impact / 1e3,
    }


def fmt_energy(mt: float) -> str:
    if mt < 1e-3:
        return f"{mt * 1e6:,.0f} t TNT"
    if mt < 1:
        return f"{mt * 1e3:,.1f} kt TNT"
    return f"{mt:,.0f} Mt TNT"
