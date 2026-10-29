"""Forecast-driven collection planning (multi-trip trucks, wards larger than one truckload).

Inputs come from data/planning_inputs.csv (depot, trucks, truck capacity, zone storage days).
Assumptions NOT in that file (change them below if you know better):
  shift_hours, speed_kmh, stop_min, unload_min, road_factor, and that trucks unload at the depot
  (dump_extra_km adds a round-trip distance to a dump site per trip; 0 = unload at the depot).

Policies compared over the test period:
  daily      - every ward collected every day
  alternate  - every ward collected on alternate days
  forecast   - ward collected if (current fill + MODEL forecast for tomorrow) reaches RISK_THRESHOLD of its storage
  naive      - same rule, using today's amount as tomorrow's forecast
  oracle     - same rule, using tomorrow's true amount (best possible forecast, for reference)
Waste above a ward's storage capacity is counted as overflow (it stays and is collected later).
"""
import math
import numpy as np
import pandas as pd

CFG = dict(depot=(27.188, 78.015), trucks=82, capacity_kg=4220.0, storage_days=1.9,
           risk_threshold=0.90, shift_hours=8.0, speed_kmh=20.0, stop_min=20.0, unload_min=15.0,
           road_factor=1.3, dump_extra_km=0.0)


def load_inputs(path="data/planning_inputs.csv"):
    t = pd.read_csv(path).set_index("input")["value"]
    CFG.update(depot=(float(t["depot_lat"]), float(t["depot_lon"])), trucks=int(round(t["trucks_per_day"])),
               capacity_kg=float(t["truck_capacity_kg"]), storage_days=float(t["zone_storage_days"]))
    return dict(CFG)


def haversine(a, b):
    R = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def plan_day(loads, names, depot_d, pair_d, fill_ratio):
    """loads: {zone_idx: kg to collect}. Splits each ward into truckloads, packs them into trips
    (first-fit decreasing), orders stops by nearest neighbour, then schedules trips on trucks
    within the shift. Returns (served_trips, unserved_trips, truck_minutes_used, total_trip_minutes)."""
    cap = CFG["capacity_kg"]
    parts = []
    for z, kg in loads.items():
        full, rem = divmod(kg, cap)
        parts += [(z, cap)] * int(full)
        if rem > 1:
            parts.append((z, rem))
    parts.sort(key=lambda p: -p[1])
    bins = []
    for z, kg in parts:
        for b in bins:
            if b["left"] >= kg and z not in b["zones"]:
                b["parts"].append((z, kg)); b["zones"].add(z); b["left"] -= kg; break
        else:
            bins.append({"parts": [(z, kg)], "zones": {z}, "left": cap - kg})
    trips = []
    for b in bins:
        left, pos, order, dist = list(b["zones"]), None, [], 0.0
        while left:
            if pos is None:
                nxt = min(left, key=lambda z: depot_d[z]); dist += depot_d[nxt]
            else:
                nxt = min(left, key=lambda z: pair_d[pos, z]); dist += pair_d[pos, nxt]
            pos = nxt; order.append(nxt); left.remove(nxt)
        dist += depot_d[pos] + CFG["dump_extra_km"]
        minutes = dist / CFG["speed_kmh"] * 60 + CFG["stop_min"] * len(order) + CFG["unload_min"]
        trips.append({"stops": order, "parts": b["parts"], "distance_km": dist, "minutes": minutes,
                      "load_kg": sum(k for _, k in b["parts"]),
                      "priority": float(np.mean([fill_ratio[z] for z in order]))})
    trips.sort(key=lambda t: -t["priority"])
    remaining = np.full(CFG["trucks"], CFG["shift_hours"] * 60.0)
    served, unserved = [], []
    for t in trips:
        j = int(np.argmax(remaining))
        if remaining[j] >= t["minutes"]:
            remaining[j] -= t["minutes"]; served.append(t)
        else:
            unserved.append(t)
    used = float(CFG["trucks"] * CFG["shift_hours"] * 60.0 - remaining.sum())
    return served, unserved, int((remaining < CFG["shift_hours"] * 60.0 - 1e-9).sum()), sum(t["minutes"] for t in trips)


def simulate(test_df, pred_col, policy):
    zones = sorted(test_df["zone"].unique())
    idx = {z: i for i, z in enumerate(zones)}
    coords = [tuple(test_df.loc[test_df.zone == z, ["lat", "lon"]].iloc[0]) for z in zones]
    rf = CFG["road_factor"]
    depot_d = np.array([haversine(CFG["depot"], c) * rf for c in coords])
    pair_d = np.array([[haversine(a, b) * rf for b in coords] for a in coords])
    mean_daily = np.array([test_df.loc[test_df.zone == z, "waste_kg"].mean() for z in zones])
    capacity = CFG["storage_days"] * mean_daily
    fill = np.zeros(len(zones))  # all wards emptied just before the test period starts
    dates = sorted(test_df["date"].unique())
    actual = test_df.pivot(index="date", columns="zone", values="waste_kg")[zones]
    pred = test_df.pivot(index="date", columns="zone", values=pred_col)[zones] if pred_col else None
    log = []
    for i, d in enumerate(dates):
        fill = fill + actual.loc[d].values
        over = np.maximum(0.0, fill - capacity)
        if policy == "daily":
            visit = list(range(len(zones)))
        elif policy == "alternate":
            visit = list(range(len(zones))) if i % 2 == 0 else []
        else:
            nxt = pred.loc[dates[i + 1]].values if i + 1 < len(dates) else np.zeros(len(zones))
            visit = [k for k in range(len(zones)) if (fill[k] + nxt[k]) / capacity[k] >= CFG["risk_threshold"]]
        loads = {k: float(fill[k]) for k in visit}
        served, unserved, trucks_used, trip_minutes = plan_day(loads, zones, depot_d, pair_d, fill / capacity) \
            if visit else ([], [], 0, 0.0)
        for t in served:
            for k, kg in t["parts"]:
                fill[k] = max(0.0, fill[k] - kg)
        log.append({"date": d, "wards_visited": len({k for t in served for k in t["stops"]}),
                    "trips": len(served), "distance_km": sum(t["distance_km"] for t in served),
                    "truck_hours": sum(t["minutes"] for t in served) / 60.0,
                    "collected_kg": sum(t["load_kg"] for t in served),
                    "unserved_trips": len(unserved), "trucks_used": trucks_used,
                    "trucks_needed": int(math.ceil(trip_minutes / (CFG["shift_hours"] * 60.0))) if trip_minutes else 0,
                    "overflow_wards": int((over > 0).sum()), "overflow_kg": float(over.sum()),
                    "routes": [{"stops": [zones[k] for k in t["stops"]], "distance_km": round(t["distance_km"], 2),
                                "time_min": round(t["minutes"], 1), "load_kg": round(t["load_kg"], 0)} for t in served]})
    return pd.DataFrame(log)


def summarize(log):
    return {"total_distance_km": round(float(log["distance_km"].sum()), 0),
            "total_truck_hours": round(float(log["truck_hours"].sum()), 0),
            "total_trips": int(log["trips"].sum()),
            "overflow_ward_days": int(log["overflow_wards"].sum()),
            "overflow_tonnes": round(float(log["overflow_kg"].sum()) / 1000, 0),
            "collected_tonnes": round(float(log["collected_kg"].sum()) / 1000, 0),
            "unserved_trips": int(log["unserved_trips"].sum()),
            "mean_trucks_needed": round(float(log["trucks_needed"].mean()), 1),
            "peak_trucks_needed": int(log["trucks_needed"].max()),
            "days_over_fleet": int((log["trucks_needed"] > CFG["trucks"]).sum())}
