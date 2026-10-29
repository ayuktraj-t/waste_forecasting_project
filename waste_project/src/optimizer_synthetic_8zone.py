"""Forecast-driven collection planning: which zones to visit, how to route trucks.

Policy comparison over the test period:
  daily     - every zone collected every day (safe but wasteful fixed schedule)
  alternate - every zone collected on alternate days (cheaper fixed schedule)
  forecast - zone collected only if (current fill + forecast for next day) exceeds a risk threshold
Routing: first-fit-decreasing truck assignment + nearest-neighbour stop order from a depot.
"""
import math
import numpy as np
import pandas as pd

DEPOT = (27.490, 77.676)
TRUCKS = 5
TRUCK_CAPACITY_KG = 9000
AVG_SPEED_KMH = 25
STOP_TIME_MIN = 15
STORAGE_DAYS = 3.0        # zone storage capacity = 3 x average daily waste
RISK_THRESHOLD = 0.90     # collect if predicted fill ratio >= this


def haversine(a, b):
    R = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def plan_routes(zones_to_visit, coords, load_kg):
    """First-fit-decreasing assignment of zones to trucks (capacity), then nearest-neighbour
    ordering of stops inside each truck. Returns (routes, unserved_zones)."""
    bins, unserved = [], []
    for z in sorted(zones_to_visit, key=lambda z: -load_kg[z]):
        for bn in bins:
            if bn["load"] + load_kg[z] <= TRUCK_CAPACITY_KG:
                bn["zones"].append(z); bn["load"] += load_kg[z]; break
        else:
            if len(bins) < TRUCKS:
                bins.append({"zones": [z], "load": load_kg[z]})
            else:
                unserved.append(z)
    routes = []
    for bn in bins:
        pos, left, order, dist = DEPOT, list(bn["zones"]), [], 0.0
        while left:
            nxt = min(left, key=lambda z: haversine(pos, coords[z]))
            dist += haversine(pos, coords[nxt]); pos = coords[nxt]; order.append(nxt); left.remove(nxt)
        dist += haversine(pos, DEPOT)
        minutes = dist / AVG_SPEED_KMH * 60 + STOP_TIME_MIN * len(order)
        routes.append({"stops": order, "distance_km": round(dist, 2), "time_min": round(minutes, 1),
                       "load_kg": round(bn["load"], 0)})
    return routes, unserved


def simulate(test_df, pred_col, policy):
    """test_df: columns date, zone, lat, lon, waste_kg, <pred_col>. Returns daily log DataFrame."""
    zones = sorted(test_df["zone"].unique())
    coords = {z: tuple(test_df.loc[test_df.zone == z, ["lat", "lon"]].iloc[0]) for z in zones}
    capacity = {z: STORAGE_DAYS * test_df.loc[test_df.zone == z, "waste_kg"].mean() for z in zones}
    fill = {z: 0.3 * capacity[z] for z in zones}
    dates = sorted(test_df["date"].unique())
    actual = test_df.pivot(index="date", columns="zone", values="waste_kg")
    pred = test_df.pivot(index="date", columns="zone", values=pred_col)
    log = []
    for i, d in enumerate(dates):
        for z in zones:
            fill[z] += actual.loc[d, z]
        overflow = [z for z in zones if fill[z] > capacity[z]]
        if policy == "daily":
            visit = zones
        elif policy == "alternate":
            visit = zones if i % 2 == 0 else []
        else:  # forecast-driven: use tomorrow's forecast
            if i + 1 < len(dates):
                nxt = pred.loc[dates[i + 1]]
                visit = [z for z in zones if (fill[z] + nxt[z]) / capacity[z] >= RISK_THRESHOLD]
            else:
                visit = []
        load = {z: min(fill[z], TRUCK_CAPACITY_KG) for z in visit}
        routes, unserved = plan_routes(visit, coords, load) if visit else ([], [])
        served = [s for r in routes for s in r["stops"]]
        for z in served:
            fill[z] -= load[z]
        log.append({"date": d, "visited": len(served), "overflow_zones": len(overflow),
                    "distance_km": sum(r["distance_km"] for r in routes),
                    "time_min": sum(r["time_min"] for r in routes),
                    "collected_kg": sum(load[z] for z in served),
                    "routes": routes})
    return pd.DataFrame(log)


def summarize(log):
    return {"total_distance_km": round(log["distance_km"].sum(), 1),
            "total_time_h": round(log["time_min"].sum() / 60, 1),
            "total_stops": int(log["visited"].sum()),
            "overflow_zone_days": int(log["overflow_zones"].sum()),
            "collected_tonnes": round(log["collected_kg"].sum() / 1000, 1)}
