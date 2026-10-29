"""End-to-end pipeline on ward-level daily data. Usage: python run_all.py
Reads data/waste_data.csv and data/planning_inputs.csv, writes everything to ./outputs."""
import argparse, json, os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
os.chdir(HERE)
from features import build_features, time_split, HIST_FEATURES, ALL_FEATURES
from models import tabular_models, metrics, LSTMForecaster, HAS_TORCH, run_torch_lstm
import optimizer as opt

warnings.filterwarnings("ignore")
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/waste_data.csv")
ap.add_argument("--inputs", default="data/planning_inputs.csv")
args = ap.parse_args()
T0 = time.time()


def tick(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


# 1-3. Load, clean, features
raw = pd.read_csv(args.data, parse_dates=["date"])
tick(f"Loaded {args.data}: {raw.shape[0]:,} rows, {raw.zone.nunique()} wards, "
     f"{raw.date.min().date()} to {raw.date.max().date()}")
raw = raw.drop_duplicates(["zone", "date"]).sort_values(["zone", "date"])
raw["waste_kg"] = raw.groupby("zone")["waste_kg"].transform(lambda s: s.interpolate().clip(lower=0))
df = build_features(raw)
train, test = time_split(df, 0.2)
tick(f"Train {len(train):,} rows, test {len(test):,} rows (test from {test.date.min().date()})")

# 4. Forecasting: simple baselines first, then models
rows = []
y = test["waste_kg"]
for name, col in [("Naive: yesterday", "lag_1"), ("Naive: same weekday last week", "lag_7"),
                  ("Naive: ward 28-day average", "roll_mean_28")]:
    rows.append({"Model": name, "Features": "none", **metrics(y, test[col])})
for fset_name, feats in [("Historical-only", HIST_FEATURES), ("Context-aware", ALL_FEATURES)]:
    for name, model in tabular_models().items():
        model.fit(train[feats], train["waste_kg"])
        rows.append({"Model": name, "Features": fset_name, **metrics(y, model.predict(test[feats]))})
        tick(f"{name:18s} {fset_name:16s} MAE={rows[-1]['MAE']:.0f} MAPE={rows[-1]['MAPE']:.2f}% R2={rows[-1]['R2']:.3f}")

df_sorted = df.sort_values(["zone", "date"]).reset_index(drop=True)
is_train = df_sorted["date"] < test["date"].min()
tr = df_sorted[is_train]
res = run_torch_lstm(tr[ALL_FEATURES].values, tr["waste_kg"].values, tr["zone"].values,
                     df_sorted[ALL_FEATURES].values, df_sorted["zone"].values) if HAS_TORCH else None
if res is not None:
    lstm_name, (p_all, idx) = "LSTM", res
else:
    lstm = LSTMForecaster(window=7)
    lstm.fit(tr[ALL_FEATURES].values, tr["waste_kg"].values, tr["zone"].values)
    lstm_name = lstm.name
    p_all, idx = lstm.predict(df_sorted[ALL_FEATURES].values, df_sorted["zone"].values)
lstm_pred = pd.Series(p_all, index=idx)
test_idx = [i for i in idx if not is_train.iloc[i]]
rows.append({"Model": lstm_name, "Features": "Context-aware",
             **metrics(df_sorted.loc[test_idx, "waste_kg"].values, lstm_pred[test_idx].values)})
tick(f"{lstm_name} MAE={rows[-1]['MAE']:.0f} R2={rows[-1]['R2']:.3f}")
met = pd.DataFrame(rows)
ref_mae = float(met.loc[met.Model == "Naive: same weekday last week", "MAE"].iloc[0])
met["skill_vs_weekly_naive_pct"] = ((1 - met.MAE / ref_mae) * 100).round(1)
met.round(3).to_csv(f"{OUT}/metrics.csv", index=False)

# 5. Explainability (SHAP, XGBoost context-aware; this model is fixed in advance, not picked on the test set)
xgb = tabular_models()["XGBoost"].fit(train[ALL_FEATURES], train["waste_kg"])
sample = test[ALL_FEATURES].sample(min(1500, len(test)), random_state=0)
sv = shap.TreeExplainer(xgb).shap_values(sample)
imp = pd.DataFrame({"feature": ALL_FEATURES, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values(
    "mean_abs_shap", ascending=False)
imp.round(3).to_csv(f"{OUT}/shap_importance.csv", index=False)
plt.figure(); shap.summary_plot(sv, sample, show=False, max_display=12)
plt.tight_layout(); plt.savefig(f"{OUT}/shap_summary.png", dpi=130); plt.close()
tick("SHAP done")

# 6. Uncertainty: split-conformal 90% intervals (relative residuals), calibrated on the last 15% of training
cut = train["date"].sort_values().iloc[int(len(train) * 0.85)]
fit_part, cal = train[train.date < cut], train[train.date >= cut]
m_u = tabular_models()["XGBoost"].fit(fit_part[ALL_FEATURES], fit_part["waste_kg"])
rel = np.abs(cal["waste_kg"] - m_u.predict(cal[ALL_FEATURES])) / cal["waste_kg"]
q = float(np.quantile(rel, 0.90))
p_u = m_u.predict(test[ALL_FEATURES])
lo, hi = p_u * (1 - q), p_u * (1 + q)
coverage = float(((test["waste_kg"] >= lo) & (test["waste_kg"] <= hi)).mean())
bias_pct = float((p_u.mean() / test["waste_kg"].mean() - 1) * 100)
tick(f"Conformal 90% interval +/-{q * 100:.1f}%, test coverage {coverage * 100:.1f}%, mean bias {bias_pct:+.2f}%")

# 7. Generalization: unseen wards (10-fold by ward) and unseen year (rolling origin)
cz = []
for k, (a, b) in enumerate(GroupKFold(n_splits=10).split(df, groups=df["zone"])):
    tr_k, te_k = df.iloc[a], df.iloc[b]
    m = tabular_models()["XGBoost"].fit(tr_k[ALL_FEATURES], tr_k["waste_kg"])
    cz.append({"fold": k + 1, "wards_held_out": int(te_k.zone.nunique()),
               **metrics(te_k["waste_kg"], m.predict(te_k[ALL_FEATURES])),
               "weekly_naive_MAE": float(np.mean(np.abs(te_k["waste_kg"] - te_k["lag_7"])))})
pd.DataFrame(cz).round(3).to_csv(f"{OUT}/cross_zone.csv", index=False)
ct = []
for yr in sorted(df["date"].dt.year.unique())[1:]:
    tr_y, te_y = df[df.date.dt.year < yr], df[df.date.dt.year == yr]
    m = tabular_models()["XGBoost"].fit(tr_y[ALL_FEATURES], tr_y["waste_kg"])
    ct.append({"test_year": int(yr), **metrics(te_y["waste_kg"], m.predict(te_y[ALL_FEATURES])),
               "weekly_naive_MAE": float(np.mean(np.abs(te_y["waste_kg"] - te_y["lag_7"])))})
pd.DataFrame(ct).round(3).to_csv(f"{OUT}/cross_time.csv", index=False)
tick("Generalization done")

# Predictions, city totals, per-ward errors
pdf = test[["date", "zone", "lat", "lon", "land_use", "waste_kg", "lag_1", "lag_7"]].copy()
pdf["pred"], pdf["pred_lo"], pdf["pred_hi"] = p_u, lo, hi
pdf.round(1).to_csv(f"{OUT}/test_predictions.csv", index=False)
city = pdf.groupby("date").agg(actual_kg=("waste_kg", "sum"), pred_kg=("pred", "sum"),
                               weekly_naive_kg=("lag_7", "sum")).reset_index()
city.round(0).to_csv(f"{OUT}/city_daily.csv", index=False)
city_mape = float(np.mean(np.abs(city.pred_kg - city.actual_kg) / city.actual_kg) * 100)
city_mape_naive = float(np.mean(np.abs(city.weekly_naive_kg - city.actual_kg) / city.actual_kg) * 100)
we = pdf.groupby("zone").apply(lambda g: pd.Series({
    "mean_kg": g.waste_kg.mean(), "MAE": np.mean(np.abs(g.pred - g.waste_kg)),
    "MAPE": np.mean(np.abs(g.pred - g.waste_kg) / g.waste_kg) * 100,
    "bias_pct": (g.pred.mean() / g.waste_kg.mean() - 1) * 100,
    "inside_band_pct": ((g.waste_kg >= g.pred_lo) & (g.waste_kg <= g.pred_hi)).mean() * 100}))
we.round(2).reset_index().to_csv(f"{OUT}/ward_errors.csv", index=False)

best_ctx = met[met.Features == "Context-aware"].sort_values("MAE").iloc[0]
json.dump({"conformal_interval_pct": round(q * 100, 1), "interval_coverage_pct": round(coverage * 100, 1),
           "mean_bias_pct": round(bias_pct, 2), "torch_available": bool(res is not None),
           "train_rows": len(train), "test_rows": len(test), "wards": int(raw.zone.nunique()),
           "test_start": str(test.date.min().date()), "test_end": str(test.date.max().date()),
           "city_mape_pct": round(city_mape, 2), "city_mape_weekly_naive_pct": round(city_mape_naive, 2),
           "weekly_naive_MAE": round(ref_mae, 1), "best_context_model": best_ctx.Model,
           "best_context_MAE": round(float(best_ctx.MAE), 1)},
          open(f"{OUT}/summary.json", "w"), indent=2)
tick("Modelling done; running collection planning")
import run_planning
run_planning.main(args.inputs)
tick("DONE. Outputs in ./outputs")
