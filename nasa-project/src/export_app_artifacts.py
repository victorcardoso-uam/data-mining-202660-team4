"""Export the serialized models and the app dataset used by the Streamlit app.

Run from the repository root, after notebooks 03 to 07:

    uv run python nasa-project/src/export_app_artifacts.py

Inputs (not versioned): nasa-project/data/raw and nasa-project/data/processed.
Outputs (versioned):
    models/pha_random_forest.joblib    Random Forest from notebook 05 (reloaded, not retrained)
    models/pha_decision_tree.joblib    Decision tree, depth 5 (classical baseline)
    models/pha_neural_network.joblib   MLP classifier (ANN), same features and split
    models/diameter_rf.joblib          Physics-guided Random Forest from notebook 04
    models/kmeans_risk_profiles.joblib StandardScaler + K-Means (K = 4) from notebook 06
    models/benchmark.json              Test metrics, confusion matrices, curves, loss curves
    models/metadata.json               Features, thresholds, policy constants, slider ranges
    data/neo_catalogue.csv.gz          One row per object: features, predictions, tier
"""

import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score, roc_curve,
                             silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

REPO_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_DIR / "nasa-project"
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROC_DIR = PROJECT_DIR / "data" / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"
MODELS_DIR = REPO_DIR / "models"
DATA_DIR = REPO_DIR / "data"

RECALL_TARGET = 0.99
MLP_EPOCHS = 80
MLP_PATIENCE = 12
LD_AU = 0.00257


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"pdes": str, "designation": str}, low_memory=False, **kwargs)


def threshold_for_recall(y_true, proba, target=RECALL_TARGET) -> float:
    # Highest threshold that still catches at least `target` of the PHAs (same rule as notebook 05)
    pos = np.sort(np.asarray(proba)[np.asarray(y_true) == 1])[::-1]
    return float(pos[int(np.ceil(target * len(pos))) - 1])


def classification_metrics(y_true, proba, threshold) -> dict:
    pred = proba >= threshold
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "missed_pha": int(cm[1, 0]),
        "share_monitored": float(pred.mean()),
        "confusion_matrix": cm.tolist(),
    }


def train_mlp(prep, X, y, random_state):
    """Train the ANN epoch by epoch to record train/validation loss curves (early stopping on val loss)."""
    X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    prep = clone(prep).fit(X_fit)
    Xf, Xv = prep.transform(X_fit), prep.transform(X_val)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3,
                        learning_rate_init=1e-3, batch_size=256, shuffle=True,
                        random_state=random_state)
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_pr_auc": []}
    best, best_loss, waited = None, np.inf, 0
    for epoch in range(1, MLP_EPOCHS + 1):
        mlp.partial_fit(Xf, y_fit, classes=[0, 1])
        p_fit, p_val = mlp.predict_proba(Xf)[:, 1], mlp.predict_proba(Xv)[:, 1]
        val_loss = log_loss(y_val, p_val)
        history["epoch"].append(epoch)
        history["train_loss"].append(log_loss(y_fit, p_fit))
        history["val_loss"].append(val_loss)
        history["val_pr_auc"].append(average_precision_score(y_val, p_val))
        if val_loss < best_loss - 1e-5:
            best, best_loss, waited, best_epoch = copy.deepcopy(mlp), val_loss, 0, epoch
        else:
            waited += 1
            if waited >= MLP_PATIENCE:
                break
    history["best_epoch"] = best_epoch
    model = Pipeline([("prep", prep), ("model", best)])
    threshold = threshold_for_recall(y_val, best.predict_proba(Xv)[:, 1])
    return model, threshold, history


def main() -> None:
    for f in [PROC_DIR / "07_risk_policy" / "priority_queue.csv",
              PROC_DIR / "05_classification" / "rf_orbit_observation.joblib"]:
        if not f.exists():
            raise FileNotFoundError(f"{f} is missing: run notebooks 03 to 07 first.")
    MODELS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    config = json.loads((REPORTS_DIR / "03_preprocessing" / "preprocessing_config.json").read_text())
    rs = config["random_state"]
    cfg = config["classification"]
    num, cat = cfg["features_orbit_obs"]["numeric"], cfg["features_orbit_obs"]["categorical"]
    features = num + cat
    target = cfg["target"]

    train = read_csv(PROC_DIR / "03_preprocessing" / "classification_train.csv")
    test = read_csv(PROC_DIR / "03_preprocessing" / "classification_test.csv")
    y_train, y_test = train[target].astype(int), test[target].astype(int)
    prep = joblib.load(PROC_DIR / "03_preprocessing" / "preprocessor_orbit_obs.joblib")

    # ---- Classification models -------------------------------------------------
    rf = joblib.load(PROC_DIR / "05_classification" / "rf_orbit_observation.joblib")
    rf_thr = json.loads((REPORTS_DIR / "05_classification" / "thresholds.json").read_text())
    rf_threshold = rf_thr["thresholds"][rf_thr["chosen_model"]]

    tree = Pipeline([("prep", clone(prep)), ("model", DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=50, class_weight="balanced", random_state=rs))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    tree_oof = cross_val_predict(tree, train[features], y_train, cv=cv, method="predict_proba")[:, 1]
    tree_threshold = threshold_for_recall(y_train, tree_oof)
    tree.fit(train[features], y_train)

    print("Training the neural network…")
    mlp, mlp_threshold, history = train_mlp(prep, train[features], y_train, rs)
    print(f"  best epoch {history['best_epoch']}, val loss {min(history['val_loss']):.4f}")

    models = {
        "Random Forest": ("pha_random_forest.joblib", rf, rf_threshold),
        "Decision Tree": ("pha_decision_tree.joblib", tree, tree_threshold),
        "Neural Network (MLP)": ("pha_neural_network.joblib", mlp, mlp_threshold),
    }
    benchmark = {"classification": {}, "recall_target": RECALL_TARGET,
                 "test_size": int(len(test)), "test_pha": int(y_test.sum())}
    for name, (file, model, thr) in models.items():
        joblib.dump(model, MODELS_DIR / file, compress=3)
        proba = model.predict_proba(test[features])[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        keep = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))
        benchmark["classification"][name] = {
            "file": file,
            "operating": classification_metrics(y_test, proba, thr),
            "default_0.5": classification_metrics(y_test, proba, 0.5),
            "roc_curve": {"fpr": fpr[keep].tolist(), "tpr": tpr[keep].tolist()},
        }
    rule = ((test["q"] <= 1.05) & (test["ad"] >= 0.95)).astype(float).to_numpy()
    benchmark["classification"]["Rule: crosses Earth's orbit"] = {
        "operating": classification_metrics(y_test, rule, 0.5)}
    benchmark["nn_history"] = history

    # ---- Diameter regression (notebook 04) -------------------------------------
    joblib.dump(joblib.load(PROC_DIR / "04_diameter" / "rf_albedo_term.joblib"),
                MODELS_DIR / "diameter_rf.joblib", compress=3)
    reg = pd.read_csv(REPORTS_DIR / "04_diameter" / "test_metrics.csv", index_col=0)
    benchmark["regression"] = {m: {"r2": float(r["R²"]), "rmse_dex": float(r["RMSE (dex)"]),
                                   "pct_off_by_factor_2": float(r["% off by a factor > 2"])}
                               for m, r in reg.iterrows()}

    # ---- Clustering (notebook 06) ----------------------------------------------
    clusters = read_csv(PROC_DIR / "06_clustering" / "catalogue_clusters.csv")
    Xc = pd.DataFrame({
        "log10_diameter": np.log10(clusters["diameter_km"]),
        "log10_moid": np.log10(clusters["moid_au"] + 1e-4),
        "inclination": clusters["i"],
        "eccentricity": clusters["e"],
    })
    kmeans = Pipeline([("scaler", StandardScaler()),
                       ("kmeans", KMeans(n_clusters=4, n_init=10, random_state=rs))]).fit(Xc)
    raw_labels = kmeans.predict(Xc)
    # Map raw K-Means labels to the notebook's labels (ordered by median diameter)
    relabel = (pd.crosstab(raw_labels, clusters["cluster"]).idxmax(axis=1)).astype(int).to_dict()
    agreement = float(np.mean(pd.Series(raw_labels).map(relabel).to_numpy() == clusters["cluster"].to_numpy()))
    joblib.dump(kmeans, MODELS_DIR / "kmeans_risk_profiles.joblib", compress=3)
    sample = np.random.default_rng(rs).choice(len(Xc), size=min(10_000, len(Xc)), replace=False)
    Xs = kmeans.named_steps["scaler"].transform(Xc)
    benchmark["clustering"] = {
        "k": 4,
        "silhouette": float(silhouette_score(Xs[sample], raw_labels[sample])),
        "inertia": float(kmeans.named_steps["kmeans"].inertia_),
        "agreement_with_notebook": agreement,
        "k_selection": pd.read_csv(REPORTS_DIR / "06_clustering" / "k_selection.csv").to_dict(orient="list"),
    }
    print(f"  K-Means labels match notebook 06 for {agreement:.1%} of objects")

    # ---- App dataset: one row per object ---------------------------------------
    raw = read_csv(sorted(RAW_DIR.glob("near_earth_asteroids_2025*.csv"))[0])
    approaches = read_csv(sorted(RAW_DIR.glob("asteroid_close_approaches_2015_2035*.csv"))[0])
    approaches["date"] = pd.to_datetime(approaches["close_approach_date"])
    extraction = approaches.loc[approaches["is_future"], "date"].min().normalize()
    v_inf = approaches.groupby("designation")["velocity_infinity_km_s"].median().rename("v_inf_kms")

    raw["first_obs_year"] = pd.to_datetime(raw["first_obs"], errors="coerce").dt.year
    queue = read_csv(PROC_DIR / "07_risk_policy" / "priority_queue.csv")
    catalogue = (queue.merge(raw[["pdes", "class", "H", "a", "e", "i", "q", "ad", "condition_code",
                                  "data_arc_years", "first_obs", "first_obs_year", "albedo"]],
                             on="pdes", how="left")
                 .merge(v_inf, left_on="pdes", right_index=True, how="left"))
    catalogue["split"] = np.where(catalogue["pdes"].isin(set(test["pdes"])), "test",
                                  np.where(catalogue["pdes"].isin(set(train["pdes"])), "train", "—"))
    catalogue.to_csv(DATA_DIR / "neo_catalogue.csv.gz", index=False, compression="gzip")

    # ---- Metadata for the app ---------------------------------------------------
    interval = json.loads((REPORTS_DIR / "04_diameter" / "interval.json").read_text())
    missing = raw[features + ["H", "moid_au", "diameter_km", "albedo", "rot_per"]].isna().mean()
    q = raw[num].quantile([0.01, 0.5, 0.99])
    metadata = {
        "features": {"numeric": num, "categorical": cat, "forbidden": cfg["forbidden"]},
        "target": target,
        "models": {name: {"file": file, "threshold": thr} for name, (file, _, thr) in models.items()},
        "chosen_model": "Random Forest",
        "recall_target": RECALL_TARGET,
        "classes": sorted(raw["class"].dropna().unique().tolist()),
        "feature_ranges": {c: {"p01": float(q.loc[0.01, c]), "median": float(q.loc[0.5, c]),
                               "p99": float(q.loc[0.99, c])} for c in num},
        "missing_share_raw": missing.round(4).to_dict(),
        "raw_shape": list(raw.shape),
        "extraction_date": f"{extraction:%Y-%m-%d}",
        "diameter_interval_dex": [interval["q_low_dex"], interval["q_high_dex"]],
        "diameter_h_limit": interval["H_limit"],
        "kmeans_relabel": {str(k): v for k, v in relabel.items()},
        "kmeans_features": list(Xc.columns),
        "policy": {
            "moid_max_au": 0.05, "min_diameter_km": 0.14, "urgent_approach_ld": 20,
            "uncertain_condition_code": 7, "recent_discovery_days": 365,
            "density_kg_m3": 2600, "escape_velocity_kms": 11.2, "mt_tnt_j": 4.184e15,
            "lunar_distance_au": LD_AU,
        },
        "v_inf_median_kms": float(approaches["velocity_infinity_km_s"].median()),
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (MODELS_DIR / "benchmark.json").write_text(json.dumps(benchmark, indent=2))

    print("\nTest-set benchmark (operating threshold = 99% recall):")
    rows = {n: {k: v for k, v in b["operating"].items() if k != "confusion_matrix"}
            for n, b in benchmark["classification"].items()}
    print(pd.DataFrame(rows).T.round(3).to_string())
    for p in sorted(MODELS_DIR.iterdir()) + sorted(DATA_DIR.iterdir()):
        print(f"  {p.relative_to(REPO_DIR)}  {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
