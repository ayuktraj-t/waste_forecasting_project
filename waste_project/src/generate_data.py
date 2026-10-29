"""Synthetic zone-wise municipal waste dataset (replace with real data later).

Real-data schema expected by the rest of the pipeline (one row per zone per day):
date, zone, lat, lon, pop_density, land_use, temp_c, rain_mm, humidity, is_holiday,
event_type, waste_kg
"""
import numpy as np
import pandas as pd

ZONES = {
    # zone: (lat, lon, pop_density per km2, land_use, base_kg)
    "Z1": (27.495, 77.673, 14000, "commercial", 5200),
    "Z2": (27.488, 77.680, 9000, "residential", 3800),
    "Z3": (27.502, 77.665, 12000, "mixed", 4600),
    "Z4": (27.480, 77.690, 6000, "residential", 2700),
    "Z5": (27.510, 77.678, 3500, "industrial", 2400),
    "Z6": (27.470, 77.668, 8000, "residential", 3300),
    "Z7": (27.499, 77.700, 11000, "market", 4900),
    "Z8": (27.465, 77.685, 2500, "institutional", 1900),
}
HOLIDAYS = ["01-26", "03-14", "04-14", "08-15", "10-02", "10-21", "11-09", "12-25"]


def generate(start="2023-01-01", end="2025-12-31", seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    rows = []
    for zone, (lat, lon, dens, land, base) in ZONES.items():
        for d in dates:
            doy = d.dayofyear
            temp = 25 + 11 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 2)
            monsoon = 1.0 if 180 <= doy <= 270 else 0.15
            rain = max(0.0, rng.gamma(2, 6) * monsoon * (rng.random() < 0.45 * monsoon + 0.05))
            hum = float(np.clip(45 + 35 * monsoon + rng.normal(0, 6), 15, 100))
            hol = int(d.strftime("%m-%d") in HOLIDAYS)
            ev = "none"
            r = rng.random()
            if land in ("market", "commercial") and d.weekday() in (5, 6) and r < 0.5:
                ev = "market_day"
            elif r < 0.02:
                ev = "festival"
            elif r < 0.04:
                ev = "public_event"
            kg = base
            kg *= 1 + 0.18 * (d.weekday() >= 5) * (land in ("residential", "mixed", "market"))
            kg *= 1 - 0.30 * (d.weekday() >= 5) * (land in ("industrial", "institutional"))
            kg *= 1 + 0.004 * (temp - 25)
            kg *= 1 - 0.0025 * min(rain, 40)
            kg *= 1 + 0.22 * hol
            kg *= {"none": 1.0, "market_day": 1.25, "festival": 1.45, "public_event": 1.2}[ev]
            kg *= 1 + 0.00012 * (d - dates[0]).days  # slow growth
            kg *= rng.lognormal(0, 0.05)
            rows.append((d, zone, lat, lon, dens, land, temp, rain, hum, hol, ev, kg))
    df = pd.DataFrame(rows, columns=["date", "zone", "lat", "lon", "pop_density", "land_use",
                                     "temp_c", "rain_mm", "humidity", "is_holiday", "event_type", "waste_kg"])
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df = generate()
    df.to_csv("data/waste_data.csv", index=False)
    print(df.shape, "saved to data/waste_data.csv")
