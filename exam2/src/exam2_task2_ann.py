"""Task 2 - MLPRegressor (64, 32), ReLU, early stopping + loss curve.

Saves models/best_model.joblib and models/scaler.joblib.

Run:  python src/exam2_task2_ann.py
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from exam2_common import (FEATURES, FIG_DIR, MODELS_DIR, RANDOM_STATE,
                          adjusted_r2, load_split)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _, X_train, X_test, y_train, y_test = load_split()

    scaler = StandardScaler().fit(X_train)  # fit on train only (no leakage)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        max_iter=2000,
        random_state=RANDOM_STATE,
    ).fit(Xtr, y_train)

    pred = mlp.predict(Xte)
    r2 = r2_score(y_test, pred)
    metrics = {
        "model": "4. ANN MLP (64,32) ReLU",
        "R2": r2,
        "Adj_R2": adjusted_r2(r2, len(y_test), len(FEATURES)),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "epochs": mlp.n_iter_,
        "best_val_score_R2": float(mlp.best_validation_score_),
        "final_train_loss": float(mlp.loss_curve_[-1]),
    }
    print("=== ANN holdout test metrics ===")
    for k, v in metrics.items():
        print(f"{k:>20}: {v:.4f}" if isinstance(v, float) else f"{k:>20}: {v}")

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(mlp.loss_curve_, label="Training loss (MSE/2)", color="tab:blue")
    ax1.set(xlabel="Epoch", ylabel="Training loss",
            title="MLPRegressor (64, 32) ReLU - Loss Convergence")
    ax2 = ax1.twinx()
    ax2.plot(mlp.validation_scores_, color="tab:orange", label="Validation R² (early stopping)")
    ax2.set_ylabel("Validation R²")
    best_epoch = int(np.argmax(mlp.validation_scores_))
    ax2.axvline(best_epoch, color="green", ls="--", label=f"Best epoch = {best_epoch}")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "task2_loss_curve.png", dpi=130)
    plt.close(fig)

    joblib.dump(mlp, MODELS_DIR / "best_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump({
        "metrics": metrics,
        "loss_curve": list(mlp.loss_curve_),
        "validation_scores": list(mlp.validation_scores_),
        "y_test": y_test.values,
        "y_pred": pred,
        "features": FEATURES,
    }, MODELS_DIR / "ann_results.joblib")
    print("\nSaved models/best_model.joblib, models/scaler.joblib, models/ann_results.joblib")
    print(f"Saved loss curve to {FIG_DIR / 'task2_loss_curve.png'}")


if __name__ == "__main__":
    main()
