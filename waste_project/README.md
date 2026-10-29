# Ward-level waste forecasting and collection planning

Runs on `data/waste_data.csv` (100 wards x 1,096 days, 2023-01-01 to 2025-12-31) and
`data/planning_inputs.csv` (depot, trucks, truck capacity, storage days).

## Run
```
pip install numpy pandas scikit-learn xgboost shap matplotlib streamlit plotly torch
python run_all.py            # forecasting + planning -> ./outputs  (about 4-5 minutes)
streamlit run dashboard.py   # dashboard (light theme set in .streamlit/config.toml)
```
Without `torch` the "LSTM" row is a small neural-net stand-in and is labelled as such.
Install torch and re-run to get the real LSTM.

## Where things are
| Part | File |
|---|---|
| Features (lags, rolling means, calendar, weather, location, events) | `src/features.py` |
| Models (LR, DT, RF, XGBoost, LSTM) | `src/models.py` |
| Pipeline, baselines, SHAP, 90% intervals, unseen-ward and unseen-year tests | `run_all.py` |
| Collection simulation (5 policies, sensitivity) | `src/optimizer.py`, `run_planning.py` |
| Dashboard | `dashboard.py` |

## Main results (test = last 20% of days, 2025-06-01 to 2025-12-31)
- Best model: Random Forest with context, 510 kg average error per ward-day, 41% below the "same weekday last week" rule (858 kg).
- Whole-city daily error 3.8% vs 7.2% for the weekly rule.
- 90% interval (+/-14.4%) covers 91% of test days.
- Collection: forecast-driven plan has no overflow and drives ~1% less than daily collection at 1.9 days of storage. Using yesterday's amount as the forecast overflows 608 ward-days. Savings grow with more storage (see `outputs/sensitivity_storage.csv`).

## Assumptions to confirm (not in your files)
- Depot location is the placeholder from `planning_inputs.csv`.
- 8 h shift, 20 km/h, 20 min per stop, 15 min unloading, road distance = 1.3 x straight line, trucks unload at the depot (`dump_extra_km` tests a farther dump site).
- Same truck capacity (4,220 kg) and storage (1.9 days) for every ward and truck.
- Weather and holiday values for a forecast day are the recorded ones (a perfect weather forecast).
- Average trucks needed per day ranges from about 19 to 70 (peak days up to 94) depending on speed, stop time and dump-site distance, so 82 trucks cannot be called too many or too few. With the dump at the depot it is about 27 per day.

## Data caveat
The data are unusually complete, festival days are not higher than normal days, and population density
does not relate to waste. Confirm where it came from before presenting it as real measurement.
Some days drop sharply (for example 2025-11-02: 315 t vs ~650 t typical, with no holiday or event flagged);
the model cannot predict these and they drive the largest city-level misses. They look like missing
collection or recording gaps and are worth checking.
