"""Dashboard. Run: streamlit run dashboard.py   (after: python run_all.py)"""
import json, os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

OUT = "outputs"
TEAL, ORANGE, SLATE, GREY, RED, INDIGO = "#5fcfbd", "#ef9a4d", "#475569", "#cbd5e1", "#dc2626", "#6366f1"
st.set_page_config(page_title="Waste Generation Forecasting & Intelligent Collection Optimization", layout="wide")
if not os.path.exists(f"{OUT}/summary.json") or "planning" not in json.load(open(f"{OUT}/summary.json")):
    st.error("Run `python run_all.py` first to generate the outputs."); st.stop()


def show(fig, height=430):
    fig.update_layout(template="plotly_white", height=height, margin=dict(t=90, b=40, l=10, r=10),
                      font=dict(size=13), title=dict(y=0.97, yanchor="top"),
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0))
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def load():
    return dict(
        summary=json.load(open(f"{OUT}/summary.json")), met=pd.read_csv(f"{OUT}/metrics.csv"),
        pred=pd.read_csv(f"{OUT}/test_predictions.csv", parse_dates=["date"]),
        city=pd.read_csv(f"{OUT}/city_daily.csv", parse_dates=["date"]), we=pd.read_csv(f"{OUT}/ward_errors.csv"),
        shap=pd.read_csv(f"{OUT}/shap_importance.csv"), cz=pd.read_csv(f"{OUT}/cross_zone.csv"),
        ct=pd.read_csv(f"{OUT}/cross_time.csv"), sens=pd.read_csv(f"{OUT}/sensitivity_storage.csv"),
        fleet=pd.read_csv(f"{OUT}/sensitivity_fleet.csv"),
        daily=pd.read_csv(f"{OUT}/plan_daily_daily.csv", parse_dates=["date"]),
        fc=pd.read_csv(f"{OUT}/plan_forecast_daily.csv", parse_dates=["date"]),
        routes=json.load(open(f"{OUT}/sample_route_plan.json")))


D = load()
S, met, pred, city, we = D["summary"], D["met"], D["pred"], D["city"], D["we"]
P, inp = S["planning"], S["planning_inputs"]
FRIENDLY = {"roll_mean_28": "Ward's 28-day average", "roll_mean_7": "Ward's 7-day average", "lag_1": "Yesterday's amount",
            "lag_7": "Same day last week", "lag_14": "Same day two weeks ago", "dow": "Day of week",
            "is_weekend": "Weekend", "month": "Month", "doy_sin": "Season (cycle, part 1)",
            "doy_cos": "Season (cycle, part 2)", "is_holiday": "Public holiday", "temp_c": "Temperature",
            "rain_mm": "Rainfall", "humidity": "Humidity", "pop_density": "Population density",
            "lat": "Ward latitude", "lon": "Ward longitude", "land_use_code": "Land use",
            "ev_market_day": "Market day", "ev_festival": "Festival", "ev_public_event": "Public event"}
POLICY = {"daily": ("Every day", GREY), "alternate": ("Every other day", "#94a3b8"),
          "forecast": ("Forecast-driven (model)", TEAL), "naive": ("Forecast-driven (yesterday's value)", ORANGE),
          "oracle": ("Forecast-driven (perfect forecast)", INDIGO)}

st.title("Waste Generation Forecasting & Intelligent Collection Optimization")
st.caption(f"{S['wards']} wards, test period {S['test_start']} to {S['test_end']} (the last 20% of days; "
           "the model never saw them when it was trained).")
tabs = st.tabs(["Forecast", "Models", "Explainability", "Generalization", "Collection plan"])

best = met[met.Features == "Context-aware"].sort_values("MAE").iloc[0]
naive_w = S["weekly_naive_MAE"]
skill = (1 - best.MAE / naive_w) * 100

# ------------------------------------------------------------------ Ward forecast
with tabs[0]:
    a, b = st.columns([1, 2])
    txt = a.text_input("Zone (type a number 1 to 100, e.g. 1)", value="1")
    window = b.radio("Time window", ["Last 30 days", "Last 60 days", "Full test period"], horizontal=True)
    try:
        zone = f"W{int(txt):03d}"
        assert zone in set(pred.zone)
    except Exception:
        st.warning("Type a zone number between 1 and 100; showing zone 1."); txt, zone = "1", "W001"
    z = pred[pred.zone == zone].sort_values("date")
    meta = z.iloc[0]
    if window != "Full test period":
        z = z[z.date >= z.date.max() - pd.Timedelta(days=int(window.split()[1]) - 1)]
    z = z.assign(outside=(z.waste_kg < z.pred_lo) | (z.waste_kg > z.pred_hi))
    k = st.columns(4)
    k[0].metric("Avg waste / day", f"{z.waste_kg.mean():,.0f} kg")
    k[1].metric("Avg error (MAE)", f"{(z.pred - z.waste_kg).abs().mean():,.0f} kg")
    k[2].metric("Avg error (MAPE)", f"{((z.pred - z.waste_kg).abs() / z.waste_kg).mean() * 100:.1f}%")
    k[3].metric("Actuals inside 90% band", f"{100 * (1 - z.outside.mean()):.0f}%")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z.date, y=z.pred_hi, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=z.date, y=z.pred_lo, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(148,163,184,0.25)", name="90% band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=z.date, y=z.waste_kg, mode="lines+markers", name="Actual", line=dict(color=TEAL, width=2.5),
                             marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=z.date, y=z.pred, mode="lines", name="Forecast", line=dict(color=ORANGE, width=2.5, dash="dash")))
    o = z[z.outside]
    fig.add_trace(go.Scatter(x=o.date, y=o.waste_kg, mode="markers", name="Outside band",
                             marker=dict(color=RED, size=11, symbol="x", line=dict(width=2))))
    fig.update_layout(title=f"Zone {zone}: actual vs forecast waste ({meta.land_use})", yaxis_title="kg per day",
                      hovermode="x unified")
    fig.update_xaxes(tickformat="%d %b", showgrid=False)
    show(fig, 440)
    st.caption(f"Orange dashed = model forecast, teal = what actually happened, grey band = 90% interval "
               f"(±{S['conformal_interval_pct']}%, test coverage {S['interval_coverage_pct']}%). Red × marks days the actual waste fell outside the band.")
    l, r = st.columns(2)
    lo_v, hi_v = float(min(z.waste_kg.min(), z.pred.min())), float(max(z.waste_kg.max(), z.pred.max()))
    sc = go.Figure()
    sc.add_trace(go.Scatter(x=[lo_v, hi_v], y=[lo_v, hi_v], mode="lines", name="Perfect forecast",
                            line=dict(color=GREY, dash="dot", width=2)))
    sc.add_trace(go.Scatter(x=z.waste_kg, y=z.pred, mode="markers", name="Days", marker=dict(color=TEAL, size=8, opacity=0.8),
                            customdata=z.date.dt.strftime("%d %b"),
                            hovertemplate="%{customdata}<br>Actual %{x:,.0f}<br>Forecast %{y:,.0f}<extra></extra>"))
    sc.update_layout(title="Forecast accuracy (closer to the dotted line = better)", xaxis_title="Actual kg",
                     yaxis_title="Forecast kg", showlegend=False)
    with l:
        show(sc, 360)
    zz = pred[pred.zone == zone].copy()
    zz["Day"] = zz.date.dt.day_name().str[:3]
    wk = zz.groupby("Day")[["waste_kg", "pred"]].mean().reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]).reset_index()
    bar = go.Figure()
    bar.add_trace(go.Bar(x=wk.Day, y=wk.waste_kg, name="Actual", marker_color=TEAL))
    bar.add_trace(go.Bar(x=wk.Day, y=wk.pred, name="Forecast", marker_color=ORANGE))
    bar.update_layout(title="Average waste by day of week", barmode="group", yaxis_title="kg per day")
    with r:
        show(bar, 360)

# ------------------------------------------------------------------ Overview
with tabs[0]:
    st.subheader('Whole city')
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("City total: average error", f"{S['city_mape_pct']:.1f}%",
              delta=f"{S['city_mape_pct'] - S['city_mape_weekly_naive_pct']:.1f} pts vs last-week rule", delta_color="inverse",
              help=f"'Same weekday last week' gives {S['city_mape_weekly_naive_pct']:.1f}%")
    k2.metric("Single ward: average error", f"{best.MAE:,.0f} kg", delta=f"{-skill:.0f}% vs last-week rule",
              delta_color="inverse", help=f"{best.Model}; last-week rule: {naive_w:,.0f} kg")
    k3.metric("90% band contains the actual", f"{S['interval_coverage_pct']:.0f}%",
              help=f"Band is ±{S['conformal_interval_pct']}% of the forecast")
    k4.metric("Average bias", f"{S['mean_bias_pct']:+.1f}%", help="Positive = forecasts slightly high on average")
    c = city.assign(actual=city.actual_kg / 1000, model=city.pred_kg / 1000, naive=city.weekly_naive_kg / 1000)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c.date, y=c.naive, name="Rule: same weekday last week", mode="lines",
                             line=dict(color=GREY, width=1.5)))
    fig.add_trace(go.Scatter(x=c.date, y=c.actual, name="Actual", mode="lines", line=dict(color=TEAL, width=2.5)))
    fig.add_trace(go.Scatter(x=c.date, y=c.model, name="Model forecast", mode="lines",
                             line=dict(color=ORANGE, width=2.5, dash="dash")))
    fig.update_layout(title="Whole-city waste per day: actual vs forecast", yaxis_title="Tonnes per day", hovermode="x unified")
    fig.update_xaxes(tickformat="%d %b")
    show(fig, 440)
    st.caption(f"The model follows the weekly rhythm and the dips on public holidays much more closely than the simple rule "
               f"(city error {S['city_mape_pct']:.1f}% vs {S['city_mape_weekly_naive_pct']:.1f}%).")

# ------------------------------------------------------------------ Models
with tabs[1]:
    st.subheader("How much better than simple rules?")
    m = met.copy()
    m["Model"] = m.Model.replace({"LSTM (MLP fallback - install torch)": "Neural net (stand-in for LSTM)"})
    m["label"] = np.where(m.Features == "none", m.Model,
                          m.Model + np.where(m.Features == "Context-aware", " (with context)", " (history only)"))
    m["group"] = np.where(m.Features == "none", "Simple rule",
                          np.where(m.Features == "Context-aware", "With context", "History only"))
    m = m.sort_values("MAE", ascending=False)
    fig = px.bar(m, x="MAE", y="label", color="group", orientation="h", text=m.MAE.round(0).astype(int),
                 color_discrete_map={"Simple rule": GREY, "History only": TEAL, "With context": ORANGE},
                 labels={"MAE": "Average error per ward-day (kg), lower is better", "label": "", "group": ""})
    fig.update_yaxes(categoryorder="array", categoryarray=list(m.label))
    fig.update_traces(textposition="outside")
    show(fig, 520)
    hist_best = met[met.Features == "Historical-only"].MAE.min()
    st.caption(f"Best model: {best.Model} with context, {best.MAE:,.0f} kg average error, {skill:.0f}% lower than 'same weekday "
               f"last week' ({naive_w:,.0f} kg). Adding weather, holidays, events and location lowers the error from "
               f"{hist_best:,.0f} kg (history only) to {best.MAE:,.0f} kg ({(1 - best.MAE / hist_best) * 100:.0f}% lower).")
    if not S["torch_available"]:
        st.warning("PyTorch was not installed when this ran, so the 'neural net' row is a simple stand-in, not a real LSTM. "
                   "Run `pip install torch` and `python run_all.py` to get the real LSTM result.")
    st.dataframe(met.rename(columns={"skill_vs_weekly_naive_pct": "Error lower than last-week rule (%)"}).round(2),
                 hide_index=True)

# ------------------------------------------------------------------ Drivers
with tabs[2]:
    st.subheader("What the model relies on")
    sh = D["shap"].head(10).iloc[::-1].assign(name=lambda x: x.feature.map(FRIENDLY).fillna(x.feature))
    fig = px.bar(sh, x="mean_abs_shap", y="name", orientation="h", color_discrete_sequence=[TEAL],
                 labels={"mean_abs_shap": "Average influence on the forecast (kg)", "name": ""})
    show(fig, 440)
    top = D["shap"].feature.map(FRIENDLY).fillna(D["shap"].feature).tolist()
    st.caption(f"Most influential: {top[0]}, then {top[1]} and {top[2]}. These show what the model uses, "
               "not proof that they cause waste to change.")
    with st.expander("Detailed view (SHAP): how high and low values push the forecast"):
        st.image(f"{OUT}/shap_summary.png")

# ------------------------------------------------------------------ Reliability
with tabs[3]:
    st.subheader("Does it hold up on wards and years it has not seen?")
    cz, ct = D["cz"], D["ct"]
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cz.fold.astype(str), y=cz.weekly_naive_MAE, name="Rule: same weekday last week", marker_color=GREY))
        fig.add_trace(go.Bar(x=cz.fold.astype(str), y=cz.MAE, name="Model", marker_color=TEAL))
        fig.update_layout(title="10 groups of 10 unseen wards", barmode="group", xaxis_title="Group held out",
                          yaxis_title="Average error (kg)")
        show(fig, 380)
        st.caption(f"Model error {cz.MAE.mean():,.0f} kg vs {cz.weekly_naive_MAE.mean():,.0f} kg for the rule, "
                   "on wards that were left out of training.")
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ct.test_year.astype(str), y=ct.weekly_naive_MAE, name="Rule: same weekday last week", marker_color=GREY))
        fig.add_trace(go.Bar(x=ct.test_year.astype(str), y=ct.MAE, name="Model", marker_color=TEAL))
        fig.update_layout(title="Trained on earlier years only", barmode="group", xaxis_title="Year predicted",
                          yaxis_title="Average error (kg)")
        show(fig, 380)
        st.caption("Each year is predicted by a model trained only on the years before it.")
    fig = px.histogram(we, x="MAPE", nbins=20, color_discrete_sequence=[TEAL],
                       labels={"MAPE": "Average error of the ward (%)", "count": "Wards"})
    fig.update_layout(title="Forecast error across the 100 wards", yaxis_title="Number of wards")
    show(fig, 340)
    st.caption(f"Ward errors range from {we.MAPE.min():.1f}% to {we.MAPE.max():.1f}% (median {we.MAPE.median():.1f}%). "
               f"The 90% band contained {we.inside_band_pct.min():.0f}% to {we.inside_band_pct.max():.0f}% of days across wards.")

# ------------------------------------------------------------------ Collection plan
with tabs[4]:
    st.subheader("Does forecasting help the collection plan?")
    sd = st.radio("How many days of waste can a ward store before it overflows?", [1.4, 1.9, 2.5, 3.2], index=1,
                  horizontal=True, help="Your inputs file gives a median of 1.9 and a range of 1.4 to 3.2.")
    sn = D["sens"][D["sens"].storage_days == sd].set_index("policy").loc[list(POLICY)]
    names = [POLICY[p][0] for p in sn.index]
    cols = [POLICY[p][1] for p in sn.index]
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=names, y=sn.total_distance_km / 1000, marker_color=cols,
                               text=(sn.total_distance_km / 1000).round(0).astype(int), textposition="outside"))
        fig.update_layout(title="Distance driven over the test period", yaxis_title="Thousand km", xaxis_tickangle=-25)
        show(fig, 430)
    with c2:
        fig = go.Figure(go.Bar(x=names, y=sn.overflow_ward_days, marker_color=cols,
                               text=sn.overflow_ward_days.astype(int), textposition="outside"))
        fig.update_layout(title="Overflow: ward-days over storage capacity", yaxis_title="Ward-days (of %s)" % f"{S['test_rows']:,}",
                          xaxis_tickangle=-25)
        show(fig, 430)
    dly, fcs, alt, nv = sn.loc["daily"], sn.loc["forecast"], sn.loc["alternate"], sn.loc["naive"]
    st.caption(f"With {sd} days of storage: forecast-driven collection drives {(1 - fcs.total_distance_km / dly.total_distance_km) * 100:.1f}% "
               f"less than daily collection with {int(fcs.overflow_ward_days)} overflow ward-days. Collecting every other day drives "
               f"{(1 - alt.total_distance_km / dly.total_distance_km) * 100:.1f}% less but overflows {int(alt.overflow_ward_days):,} ward-days. "
               f"Using yesterday's amount as the forecast overflows {int(nv.overflow_ward_days):,} ward-days.")
    st.info("The saving from forecasting is small when wards can store only about 2 days of waste, and grows as storage grows. "
            "Its main value is making it safe to skip visits.")

    st.subheader("How many trucks are needed?")
    dd, ff = D["daily"], D["fc"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.date, y=dd.trucks_needed, name="Collect every day", line=dict(color=GREY, width=2)))
    fig.add_trace(go.Scatter(x=ff.date, y=ff.trucks_needed, name="Forecast-driven", line=dict(color=TEAL, width=2)))
    fig.add_hline(y=inp["trucks"], line_dash="dash", line_color=RED, annotation_text=f"{inp['trucks']} trucks available",
                  annotation_position="top left")
    fig.update_layout(title="Trucks needed per day (storage 1.9 days)", yaxis_title="Trucks", yaxis_range=[0, inp["trucks"] * 1.1],
                      hovermode="x unified")
    fig.update_xaxes(tickformat="%d %b")
    show(fig, 400)
    dump = st.radio("Extra round trip to the dump site, per trip (km)", [0, 10, 20], horizontal=True,
                    help="0 means trucks unload at the depot, which is what the main results assume.")
    fl = D["fleet"][D["fleet"].dump_extra_km == dump]
    hm = fl.pivot(index="stop_min", columns="speed_kmh", values="mean_trucks_needed")
    fig = px.imshow(hm, text_auto=".0f", aspect="auto", color_continuous_scale=["#e6f4f1", TEAL],
                    labels=dict(x="Average speed (km/h)", y="Minutes per stop", color="Trucks"))
    fig.update_layout(title="Average trucks needed under different assumptions", coloraxis_showscale=False)
    show(fig, 320)
    st.caption(f"Compare with the {inp['trucks']} trucks available. The answer depends mostly on assumptions the data does not contain: "
               "speed, time per stop and how far the dump site is. Treat it as a range, not a figure.")
    st.subheader("Sample plan for the last day")
    rt = pd.DataFrame(D["routes"]).head(12)
    rt["wards"] = rt.stops.apply(lambda s: ", ".join(s))
    st.dataframe(rt[["wards", "load_kg", "distance_km", "time_min"]].rename(columns={
        "wards": "Wards on this trip", "load_kg": "Load (kg)", "distance_km": "Distance (km)", "time_min": "Time (min)"}),
        hide_index=True)

# ---------------------------------------------------------------- Limits
with tabs[4]:
    with st.expander('Data notes and assumptions (read before drawing conclusions)'):
        st.markdown(
            "- **Check the data source.** The ward file is unusually complete (100 wards x 1,096 days with no gaps) and some "
            "relationships are unexpected: festival days are not higher than normal days and population density is unrelated to waste. "
            "Confirm how the data was produced before presenting it as measured records.\n"
            "- **Weather and holiday values for the forecast day are the actual ones**, as if the weather forecast were perfect. "
            "With real weather forecasts the accuracy would be slightly lower.\n"
            "- **Test period:** one split (the last 20% of days) plus the year-by-year and unseen-ward checks.\n"
            "- **Plan assumptions not in the inputs file:** 8-hour shift, 20 km/h speed, 20 minutes per stop, 15 minutes to unload, "
            "road distance 1.3x straight-line, unloading at the depot. The depot location is marked as an approximate placeholder.\n"
            "- **The same truck capacity (4,220 kg) and the same storage (1.9 days) are used for every truck and ward**, while the "
            "inputs file gives ranges (2,020 to 9,450 kg and 1.4 to 3.2 days).\n"
            "- **The fleet of 82 is a daily average of available trucks.** The number of trucks needed is very sensitive to the assumptions above.\n"
            "- **R² looks modest (about 0.6)** because most day-to-day variation within a ward is not predictable; average error in kg "
            "and % and the comparison with simple rules are the better guide.")
