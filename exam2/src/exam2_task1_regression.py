"""Task 1 - Regression model comparison & residual diagnostics.

Models: (1) baseline OLS, (2) OLS on log(y), (3) polynomial degree 2.
Metrics on the 20% holdout: R2, adjusted R2, MAE, RMSE.
Diagnostics: 4-panel classical OLS dashboard per model
(residuals vs fitted, normal Q-Q, scale-location, residuals vs leverage).

Run:  python src/exam2_task1_regression.py
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from exam2_common import (FEATURES, FIG_DIR, MODELS_DIR, TARGET, adjusted_r2,
                          load_split)


def evaluate(name, y_true, y_pred, p):
    r2 = r2_score(y_true, y_pred)
    return {
        "model": name,
        "n_predictors": p,
        "R2": r2,
        "Adj_R2": adjusted_r2(r2, len(y_true), p),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def leverage(X_design):
    """Diagonal of the hat matrix H = X (X'X)^-1 X' (X includes intercept)."""
    Xd = np.column_stack([np.ones(len(X_design)), X_design])
    xtx_inv = np.linalg.pinv(Xd.T @ Xd)
    return np.einsum("ij,jk,ik->i", Xd, xtx_inv, Xd), Xd.shape[1]


def diagnostic_dashboard(title, fitted, resid, X_design, path):
    """Classical 4-panel OLS residual dashboard (training residuals)."""
    h, k = leverage(X_design)
    n = len(resid)
    sigma = np.sqrt(np.sum(resid**2) / (n - k))
    std_resid = resid / (sigma * np.sqrt(np.clip(1 - h, 1e-12, None)))
    cooks = std_resid**2 * h / (k * (1 - h))

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. Linearity / homoscedasticity
    a = ax[0, 0]
    a.scatter(fitted, resid, s=14, alpha=0.6)
    a.axhline(0, color="red", ls="--")
    a.set(title="1. Residuals vs Fitted (linearity)", xlabel="Fitted", ylabel="Residual")

    # 2. Normality
    a = ax[0, 1]
    (osm, osr), (slope, intercept, _) = stats.probplot(std_resid, dist="norm")
    a.scatter(osm, osr, s=14, alpha=0.6)
    a.plot(osm, slope * osm + intercept, "r--")
    a.set(title="2. Normal Q-Q (normality)", xlabel="Theoretical quantiles",
          ylabel="Standardized residual")

    # 3. Equal variance
    a = ax[1, 0]
    a.scatter(fitted, np.sqrt(np.abs(std_resid)), s=14, alpha=0.6)
    a.set(title="3. Scale-Location (homoscedasticity)", xlabel="Fitted",
          ylabel="sqrt(|standardized residual|)")

    # 4. Influence
    a = ax[1, 1]
    sc = a.scatter(h, std_resid, c=cooks, cmap="viridis", s=14, alpha=0.7)
    fig.colorbar(sc, ax=a, label="Cook's distance")
    a.axhline(0, color="grey", lw=0.8)
    a.axvline(2 * k / n, color="orange", ls="--", label=f"2k/n = {2*k/n:.3f}")
    a.legend()
    a.set(title="4. Residuals vs Leverage (influence)", xlabel="Leverage",
          ylabel="Standardized residual")

    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

    _, shapiro_p = stats.shapiro(std_resid)
    return {
        "shapiro_p": float(shapiro_p),
        "n_high_leverage": int((h > 2 * k / n).sum()),
        "n_cooks_gt_4n": int((cooks > 4 / n).sum()),
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df, X_train, X_test, y_train, y_test = load_split()
    print(f"Loaded {len(df)} rows | train={len(X_train)} test={len(X_test)}")
    capped = (df[TARGET] == df[TARGET].max()).mean()
    print(f"NOTE: {capped:.0%} of {TARGET} values sit exactly at the max "
          f"({df[TARGET].max()}) -> target is right-censored / capped.\n")

    results, diagnostics, preds = [], {}, {}

    # (1) Baseline OLS
    ols = LinearRegression().fit(X_train, y_train)
    pred = ols.predict(X_test)
    results.append(evaluate("1. Baseline OLS", y_test, pred, len(FEATURES)))
    preds["1. Baseline OLS"] = pred
    diagnostics["1. Baseline OLS"] = diagnostic_dashboard(
        "Baseline OLS - Residual Diagnostics (train)",
        ols.predict(X_train), (y_train - ols.predict(X_train)).values,
        X_train.values, FIG_DIR / "task1_diag_baseline_ols.png")

    # (2) Log-target OLS (evaluated back on original scale)
    log_ols = LinearRegression().fit(X_train, np.log(y_train))
    pred = np.exp(log_ols.predict(X_test))
    results.append(evaluate("2. Log-Target OLS", y_test, pred, len(FEATURES)))
    preds["2. Log-Target OLS"] = pred
    fit_log = log_ols.predict(X_train)
    diagnostics["2. Log-Target OLS"] = diagnostic_dashboard(
        "Log-Target OLS - Residual Diagnostics (train, log scale)",
        fit_log, (np.log(y_train) - fit_log).values,
        X_train.values, FIG_DIR / "task1_diag_log_ols.png")

    # (3) Polynomial degree 2 (scaled first to keep X'X well conditioned)
    poly = make_pipeline(StandardScaler(),
                         PolynomialFeatures(degree=2, include_bias=False),
                         LinearRegression()).fit(X_train, y_train)
    pred = poly.predict(X_test)
    p_poly = poly[1].n_output_features_
    results.append(evaluate("3. Polynomial (deg 2)", y_test, pred, p_poly))
    preds["3. Polynomial (deg 2)"] = pred
    Z_train = poly[1].transform(poly[0].transform(X_train))
    diagnostics["3. Polynomial (deg 2)"] = diagnostic_dashboard(
        "Polynomial (deg 2) - Residual Diagnostics (train)",
        poly.predict(X_train), (y_train - poly.predict(X_train)).values,
        Z_train, FIG_DIR / "task1_diag_poly2.png")

    metrics = pd.DataFrame(results).set_index("model")
    diag = pd.DataFrame(diagnostics).T.astype({"shapiro_p": float, "n_high_leverage": int, "n_cooks_gt_4n": int})
    pd.set_option("display.width", 120)
    print("=== Holdout test metrics (20%) ===")
    print(metrics.round(4))
    print("\n=== Residual health checks (train residuals) ===")
    print(diag)
    best = metrics["RMSE"].idxmin()
    print(f"\nBest regression model by test RMSE: {best}")

    coefs = pd.Series(ols.coef_, index=FEATURES)
    print("\nBaseline OLS coefficients:\n", coefs.round(4))

    joblib.dump({
        "metrics": metrics,
        "diagnostics": diag,
        "best_model": best,
        "ols_coefficients": coefs,
        "y_test": y_test.values,
        "test_predictions": preds,
        "figures": {
            "1. Baseline OLS": "reports/figures/task1_diag_baseline_ols.png",
            "2. Log-Target OLS": "reports/figures/task1_diag_log_ols.png",
            "3. Polynomial (deg 2)": "reports/figures/task1_diag_poly2.png",
        },
        "models": {"ols": ols, "log_ols": log_ols, "poly2": poly},
    }, MODELS_DIR / "regression_results.joblib")

    # Dataset summary for the Streamlit EDA tab (app runs without data/)
    joblib.dump({
        "n_rows": len(df),
        "describe": df.describe().T,
        "corr": df.corr(),
        "head": df.head(20),
        "capped_share": float(capped),
        "feature_ranges": {f: (float(df[f].min()), float(df[f].max()),
                               float(df[f].median())) for f in FEATURES},
        "target_quantiles": df[TARGET].quantile([0.05, 0.10, 0.25, 0.5]).to_dict(),
        "target_values": df[TARGET].values,
    }, MODELS_DIR / "dataset_summary.joblib")
    print("\nSaved models/regression_results.joblib, models/dataset_summary.joblib")
    print(f"Saved 3 diagnostic dashboards to {FIG_DIR}")


if __name__ == "__main__":
    main()
