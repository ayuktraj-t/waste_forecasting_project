"""Feature engineering: temporal, spatial, contextual, plus lag features."""
import numpy as np
import pandas as pd

HIST_FEATURES = ["lag_1", "lag_7", "lag_14", "roll_mean_7", "roll_mean_28"]
TEMPORAL = ["dow", "is_weekend", "month", "doy_sin", "doy_cos", "is_holiday"]
ENVIRONMENT = ["temp_c", "rain_mm", "humidity"]
SPATIAL = ["pop_density", "lat", "lon", "land_use_code"]
EVENTS = ["ev_market_day", "ev_festival", "ev_public_event"]
CONTEXT_FEATURES = TEMPORAL + ENVIRONMENT + SPATIAL + EVENTS
ALL_FEATURES = HIST_FEATURES + CONTEXT_FEATURES

LAND_USE_CODES = {"residential": 0, "commercial": 1, "mixed": 2, "industrial": 3,
                  "market": 4, "institutional": 5}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["zone", "date"]).reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = df["date"].dt.month
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    cats = sorted(df["land_use"].unique())
    df["land_use_code"] = df["land_use"].map({c: i for i, c in enumerate(cats)})
    for e in ["market_day", "festival", "public_event"]:
        df[f"ev_{e}"] = (df["event_type"] == e).astype(int)
    g = df.groupby("zone")["waste_kg"]
    df["lag_1"] = g.shift(1)
    df["lag_7"] = g.shift(7)
    df["lag_14"] = g.shift(14)
    df["roll_mean_7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    df["roll_mean_28"] = g.transform(lambda s: s.shift(1).rolling(28).mean())
    return df.dropna(subset=HIST_FEATURES).reset_index(drop=True)


def time_split(df: pd.DataFrame, test_frac=0.2):
    """Chronological split (no leakage)."""
    cutoff = df["date"].sort_values().iloc[int(len(df) * (1 - test_frac))]
    return df[df["date"] < cutoff].copy(), df[df["date"] >= cutoff].copy()
