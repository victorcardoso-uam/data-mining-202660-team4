"""Shared config for Partial Exam 2 - Team 4 (exam2_team4_health.csv)."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "exam2_team4_health.csv"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"

TARGET = "cardiac_output_l_min"
FEATURES = [
    "heart_rate_bpm",
    "blood_pressure_sys_mmHg",
    "oxygen_saturation_pct",
    "respiratory_rate_bpm",
    "body_temp_c",
]
RANDOM_STATE = 42


def load_split():
    """Load the CSV and return the same 80/20 split for every task."""
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    return df, X_train, X_test, y_train, y_test


def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)
