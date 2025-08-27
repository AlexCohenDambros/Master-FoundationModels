#!/usr/bin/env python3
"""
moe_regressor.py

Mixture-of-Experts regressor com roteador top-k (sklearn experts + gating MLP).
- Treina múltiplos experts (sklearn pipelines).
- Treina um gating classifier que aprende, por amostra, qual expert foi o melhor
  no conjunto de validação (rótulo = índice do expert com menor erro).
- Em predição usa-se router top-k: seleciona k experts com maiores probabilidades
  (renormaliza) e retorna a soma ponderada das predições desses k experts.

Como usar:
 1) demo (sintético): python moe_regressor.py
 2) treinar com CSV: python moe_regressor.py --input-csv dados.csv --target target_col --features feat1 feat2 feat3
    - se --features não for passado, todas as colunas numéricas exceto target serão usadas.
    - por padrão faz split temporal (últimos test_size% como teste) e usa val_size% do treino como validação para treinar o gating.
Saída:
  - salva o modelo em --save-path (default: ./moe_model.joblib)
"""

import argparse
import os
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class MoERegressor:
    def __init__(self, experts: List, gating=None, top_k: int = 2, random_state: int = 42):
        """
        experts: list of sklearn estimators (or pipelines) - length = n_experts
        gating: sklearn classifier. If None, MLPClassifier(...) is used.
        top_k: number of experts to keep during prediction (router top-k).
        """
        self.experts = list(experts)
        self.n_experts = len(self.experts)
        self.gating = gating if gating is not None else MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_state
        )
        self.top_k = max(1, min(top_k, self.n_experts))
        self.random_state = random_state
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, val_size: float = 0.2, time_split: bool = False):
        """
        Fit experts and gating.
        - X, y: numpy arrays (n_samples, n_features), (n_samples,)
        - val_size: fraction of training data to use as validation for gating
        - time_split: if True, use the last val_size portion of X as validation (time series style).
                      if False, random split (train_test_split).
        Procedure:
          1) Split X-> X_train / X_val
          2) Fit each expert on X_train
          3) Evaluate each expert on X_val -> compute per-sample absolute error
          4) Label each sample in X_val with index of expert with smallest error
          5) Train gating classifier on X_val -> best_expert_idx
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if y.ndim != 1:
            y = y.ravel()

        # Split into train / val for gating supervision
        if time_split:
            n_val = int(len(X) * val_size)
            if n_val <= 0:
                raise ValueError("val_size too small for time_split")
            X_train, X_val = X[:-n_val], X[-n_val:]
            y_train, y_val = y[:-n_val], y[-n_val:]
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_size, random_state=self.random_state
            )

        # 1) Train experts on X_train
        for idx, expert in enumerate(self.experts):
            expert.fit(X_train, y_train)

        # 2) Evaluate experts on X_val
        preds = np.column_stack([expert.predict(X_val) for expert in self.experts])  # shape (n_val, n_experts)
        errors = np.abs(preds - y_val.reshape(-1, 1))  # per-sample absolute errors
        best_idx = np.argmin(errors, axis=1)  # which expert had least error per sample

        # 3) Train gating classifier to predict best_idx from X_val
        # If only one class is present (rare), use DummyClassifier for stability
        unique_classes = np.unique(best_idx)
        if unique_classes.size == 1:
            # gating will always predict that single class
            self.gating = DummyClassifier(strategy="constant", constant=unique_classes[0])
            self.gating.fit(X_val, best_idx)
        else:
            self.gating.fit(X_val, best_idx)

        self._is_fitted = True
        # store metadata
        self.feature_shape_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray, top_k: Optional[int] = None, return_proba: bool = False, verbose: bool = False):
        """
        Predict with MoE using gating probabilities and top-k routing.
        - top_k: override stored top_k if provided.
        - return_proba: if True, also return gating probabilities (shape n_samples x n_experts)
        - verbose: if True, print which experts were selected for each sample with their weights.
        """
        if not self._is_fitted:
            raise RuntimeError("MoERegressor is not fitted. Call fit(...) first.")
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        top_k = self.top_k if top_k is None else max(1, min(top_k, self.n_experts))

        # obter probabilidades do gating
        proba_raw = self.gating.predict_proba(X)
        proba = np.zeros((len(X), self.n_experts))
        for col_idx, cls in enumerate(self.gating.classes_):
            cls_int = int(cls)
            if 0 <= cls_int < self.n_experts:
                proba[:, cls_int] = proba_raw[:, col_idx]

        expert_preds = np.column_stack([expert.predict(X) for expert in self.experts])
        final_preds = np.zeros(len(X))

        for i in range(len(X)):
            p = proba[i].copy()
            topk_idx = np.argsort(p)[-top_k:][::-1]
            mask = np.zeros_like(p)
            mask[topk_idx] = 1.0
            p_masked = p * mask
            s = p_masked.sum()
            weights = p_masked / s if s != 0 else mask / mask.sum()
            final_preds[i] = np.dot(weights, expert_preds[i])

            if verbose:
                print(f"Amostra {i}: top-{top_k} experts escolhidos -> {topk_idx} com pesos {weights[topk_idx]}")

        if return_proba:
            return final_preds, proba
        return final_preds

    def save(self, path: str):
        """Save the entire instance with joblib.dump"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str):
        """Load instance saved with save(). The MoERegressor class must be available."""
        obj = joblib.load(path)
        if not isinstance(obj, MoERegressor):
            raise ValueError("Loaded object is not a MoERegressor instance.")
        return obj


# -----------------------
# Helpers: synthetic dataset and CSV loader
# -----------------------
def make_synthetic_load_series(n_hours: int = 24 * 90, seed: int = 42):
    """Generate a small synthetic hourly load series for demo (n_hours default ~90 days)."""
    rng = np.random.RandomState(seed)
    t = np.arange(n_hours)
    daily = 10 * np.sin(2 * np.pi * (t % 24) / 24)
    weekly = 5 * np.sin(2 * np.pi * (t % (24 * 7)) / (24 * 7))
    trend = 0.001 * t
    temp = 20 + 8 * np.sin(2 * np.pi * (t % 24) / 24 - 0.5) + rng.normal(scale=1.0, size=n_hours)
    base = 50 + daily + weekly + trend + rng.normal(scale=2.0, size=n_hours)
    load = base + 0.5 * temp
    df = pd.DataFrame({
        "t": t,
        "hour": t % 24,
        "dayofweek": (t // 24) % 7,
        "temp": temp,
        "load": load
    })
    # add lag features
    for lag in (1, 24, 48):
        df[f"lag_{lag}"] = df["load"].shift(lag).fillna(method="bfill")
    df["target"] = df["load"].shift(-1).fillna(method="ffill")  # next-hour target
    return df


def load_csv_dataset(path: str, target_col: str, features: Optional[List[str]] = None):
    """Load CSV using pandas and return X, y and dataframe."""
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in CSV columns: {df.columns.tolist()}")
    if features is None:
        # take numeric columns except target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [c for c in numeric_cols if c != target_col]
    else:
        for c in features:
            if c not in df.columns:
                raise ValueError(f"Feature column '{c}' not found in CSV")
    X = df[features].values
    y = df[target_col].values.ravel()
    return X, y, df, features


# -----------------------
# Main: CLI
# -----------------------
def main():
    p = argparse.ArgumentParser(description="Train and save a Mixture-of-Experts regressor (top-k router).")
    p.add_argument("--input-csv", type=str, default=None, help="Path to CSV. If omitted, uses synthetic demo dataset.")
    p.add_argument("--target", type=str, default="target", help="Name of target column in CSV.")
    p.add_argument("--features", nargs="+", default=None, help="List of feature column names. If omitted, numeric cols except target are used.")
    p.add_argument("--save-path", type=str, default="./moe_model.joblib", help="Where to save trained model.")
    p.add_argument("--top-k", type=int, default=2, help="top-k experts to use during prediction.")
    p.add_argument("--test-size", type=float, default=0.2, help="Fraction for final hold-out test set.")
    p.add_argument("--val-size", type=float, default=0.2, help="Fraction of training used as gating validation.")
    p.add_argument("--time-split", action="store_true", help="Use time-based split (last portion as test/val) instead of random shuffle.")
    p.add_argument("--random-seed", type=int, default=0, help="Random seed.")
    args = p.parse_args()

    np.random.seed(args.random_seed)

    if args.input_csv is None:
        print("No input CSV provided — using synthetic demo dataset.")
        df = make_synthetic_load_series(n_hours=24 * 90, seed=args.random_seed)
        feature_cols = ["hour", "dayofweek", "temp", "lag_1", "lag_24", "lag_48"]
        X = df[feature_cols].values
        y = df["target"].values
    else:
        print(f"Loading CSV from: {args.input_csv}")
        X, y, df, feature_cols = load_csv_dataset(args.input_csv, target_col=args.target, features=args.features)
        print(f"Using feature cols: {feature_cols}")

    # Split train/test (time-based or random)
    n_samples = len(X)
    if args.time_split:
        split_idx = int(n_samples * (1.0 - args.test_size))
        X_train_all, X_test = X[:split_idx], X[split_idx:]
        y_train_all, y_test = y[:split_idx], y[split_idx:]
    else:
        X_train_all, X_test, y_train_all, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_seed
        )

    # Define experts (pipelines)
    experts = [
        make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=args.random_seed)),
        make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=args.random_seed)),
        make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=100, random_state=args.random_seed))
    ]

    # Instantiate MoE
    gating = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=args.random_seed)
    moe = MoERegressor(experts=experts, gating=gating, top_k=args.top_k, random_state=args.random_seed)

    # Fit (internally will split train->train/val for gating)
    print("Training MoE (this trains the experts then the gating)...")
    moe.fit(X_train_all, y_train_all, val_size=args.val_size, time_split=args.time_split)

    # Evaluate on test set
    y_pred, proba = moe.predict(X_test, return_proba=True, verbose=True)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Test results -> MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Save model
    save_path = os.path.abspath(args.save_path)
    moe.save(save_path)
    print(f"Saved MoE model to: {save_path}")

    # Show a small preview of actual vs predicted
    preview_n = min(20, len(y_test))
    preview_df = pd.DataFrame({"actual": y_test[:preview_n], "predicted": y_pred[:preview_n]})
    print("\nPreview (first rows):")
    print(preview_df.head(preview_n))

    # Plot first 200 points (or less)
    try:
        nplot = min(200, len(y_test))
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(y_test[:nplot], label="actual")
        plt.plot(y_pred[:nplot], label="predicted")
        plt.title("Actual vs Predicted (test subset)")
        plt.legend()
        plt.xlabel("sample index")
        plt.ylabel("target")
        plt.savefig('plot_actual_vs_predicted.png')
        plt.tight_layout()
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
