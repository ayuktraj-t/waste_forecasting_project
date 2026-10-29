"""Baseline, ensemble and deep-learning models + metrics."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import importlib.util
import os, subprocess, sys, tempfile
HAS_TORCH = importlib.util.find_spec("torch") is not None  # torch itself is only imported in src/lstm_job.py


def run_torch_lstm(X, y, zones, X_all, zones_all, timeout=900):
    """Train the real LSTM in a separate process. Returns (predictions, row_indices) or None if it
    failed or took longer than `timeout` seconds (the caller then uses the MLP stand-in)."""
    import pandas as pd
    codes = {z: i for i, z in enumerate(pd.unique(zones_all))}
    zi, zai = np.array([codes[z] for z in zones]), np.array([codes[z] for z in zones_all])
    job = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lstm_job.py")
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npz")
        np.savez(src, X=np.asarray(X, float), y=np.asarray(y, float), zones=zi,
                 X_all=np.asarray(X_all, float), zones_all=zai)
        try:
            r = subprocess.run([sys.executable, job, src, dst], timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"      LSTM stopped after {timeout}s; using the stand-in instead", flush=True)
            return None
        if r.returncode != 0 or not os.path.exists(dst):
            print("      LSTM failed; using the stand-in instead", flush=True)
            return None
        o = np.load(dst)
        return o["pred"], o["idx"]


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {
        "MAE": mean_absolute_error(y, p),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAPE": float(np.mean(np.abs((y - p) / y)) * 100),
        "R2": r2_score(y, p),
    }


def tabular_models(seed=42):
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=seed),
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9,
                                colsample_bytree=0.9, random_state=seed, n_jobs=-1),
    }


class LSTMForecaster:
    """Stand-in used when the real LSTM (src/lstm_job.py) is unavailable: an MLP on a 7-day window."""

    def __init__(self, window=7, epochs=15, seed=42):
        self.window, self.seed = window, seed
        self.name = "LSTM (MLP fallback - install torch)"

    def _windows(self, X, y, zones):
        Xs, ys, idx = [], [], []
        w = self.window
        for z in np.unique(zones):
            ii = np.where(zones == z)[0]
            for k in range(w, len(ii)):
                Xs.append(X[ii[k - w + 1:k + 1]]); idx.append(ii[k])
                ys.append(y[ii[k]] if y is not None else 0.0)
        return np.array(Xs), np.array(ys), np.array(idx)

    def fit(self, X, y, zones):
        self.xs, self.ys = StandardScaler().fit(X), StandardScaler().fit(np.asarray(y).reshape(-1, 1))
        Xw, yw, _ = self._windows(self.xs.transform(X), self.ys.transform(np.asarray(y).reshape(-1, 1)).ravel(), zones)
        self.net = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=self.seed, early_stopping=True)
        self.net.fit(Xw.reshape(len(Xw), -1), yw)
        return self

    def predict(self, X, zones):
        Xw, _, idx = self._windows(self.xs.transform(X), None, zones)
        p = self.net.predict(Xw.reshape(len(Xw), -1))
        return self.ys.inverse_transform(np.asarray(p).reshape(-1, 1)).ravel(), idx
