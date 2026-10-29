"""Collection planning on the forecasts in outputs/test_predictions.csv. Usage: python run_planning.py
(also called at the end of run_all.py)."""
import json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
import optimizer as opt

POLICIES = [("daily", None), ("alternate", None), ("forecast", "pred"), ("naive", "naive"), ("oracle", "oracle")]


def main(inputs="data/planning_inputs.csv"):
    os.chdir(HERE)
    out = "outputs"
    t0 = time.time()
    pdf = pd.read_csv(f"{out}/test_predictions.csv", parse_dates=["date"])
    cfg = opt.load_inputs(inputs)
    plan = pdf[["date", "zone", "lat", "lon", "waste_kg"]].copy()
    plan["pred"], plan["naive"], plan["oracle"] = pdf["pred"].values, pdf["lag_1"].values, pdf["waste_kg"].values
    res, logs = {}, {}
    for policy, col in POLICIES:
        logs[policy] = opt.simulate(plan, col, policy)
        res[policy] = opt.summarize(logs[policy])
        print(f"[{time.time() - t0:4.0f}s] {policy:9s} {res[policy]}", flush=True)
        logs[policy].drop(columns="routes").to_csv(f"{out}/plan_{policy}_daily.csv", index=False, float_format="%.2f")
    json.dump(logs["forecast"].iloc[-2]["routes"], open(f"{out}/sample_route_plan.json", "w"), indent=1)

    # Sensitivity 1: how much ward storage there is (inputs file gives a range of 1.4 to 3.2 days)
    base = dict(opt.CFG)
    rows = []
    for sd in [1.4, 1.9, 2.5, 3.2]:
        opt.CFG["storage_days"] = sd
        for policy, col in POLICIES:
            rows.append({"storage_days": sd, "policy": policy, **opt.summarize(opt.simulate(plan, col, policy))})
    opt.CFG.update(base)
    pd.DataFrame(rows).to_csv(f"{out}/sensitivity_storage.csv", index=False)

    # Sensitivity 2: how many trucks are really needed depends on assumed speed, time per stop and dump-site distance
    rows = []
    for dump in [0, 10, 20]:
        for speed in [15, 20, 30]:
            for stop in [10, 20, 30]:
                opt.CFG.update(speed_kmh=speed, stop_min=stop, dump_extra_km=dump)
                s = opt.summarize(opt.simulate(plan, None, "daily"))
                rows.append({"dump_extra_km": dump, "speed_kmh": speed, "stop_min": stop,
                             "mean_trucks_needed": s["mean_trucks_needed"],
                             "peak_trucks_needed": s["peak_trucks_needed"], "days_over_fleet": s["days_over_fleet"]})
    opt.CFG.update(base)
    pd.DataFrame(rows).to_csv(f"{out}/sensitivity_fleet.csv", index=False)

    summ = json.load(open(f"{out}/summary.json")) if os.path.exists(f"{out}/summary.json") else {}
    summ["planning"] = res
    summ["planning_inputs"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
    json.dump(summ, open(f"{out}/summary.json", "w"), indent=2)
    print(f"[{time.time() - t0:4.0f}s] planning done", flush=True)


if __name__ == "__main__":
    main()
