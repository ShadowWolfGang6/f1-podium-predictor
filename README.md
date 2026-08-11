# f1-podium-predictor
A ML pipeline that predicts F1 podium finishes, feature engineering, time-based validation, calibrated probabilities, live 2026 predictions.
# F1 Podium Predictor

Predicts the probability of each driver finishing on the podium (top 3) in a Formula 1 race.

## Objective
Binary classification per driver per race. Output: podium probability (0–1).

## Metric
- Primary: Brier score
- Secondary: podium hit rate (did predicted top-3 match actual top-3)
- Baseline to beat: grid position = finishing position

## Data Source
Jolpica-F1 API via FastF1 Python library.

## How to Run
```bash
git clone https://github.com/ShadowWolfGang6/f1-podium-predictor.git
cd f1-podium-predictor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/ingest.py
python src/features.py
python src/model.py
```

## Results
**Results (Brier score, lower is better):**

| | Validation | Test |
|---|---|---|
| Grid position baseline | 0.0822 | 0.1250 |
| Logistic regression | 0.0644 | 0.0956 |

Logistic regression beat the grid-position baseline by roughly 23% on the held-out test set. TeamId was one hot encoded for this model specifically, since a linear model shouldn't treat an arbitrary label-encoded team ID as having a meaningful order. This performed marginally worse than label encoding on validation (0.0644 vs 0.0622), likely just noise given the small dataset, but one hot is the methodologically correct choice for logistic regression and is documented as the final approach.

Next step is a gradient boosted tree model (XGBoost or LightGBM), which should handle label-encoded categoricals and feature interactions more naturally than logistic regression.

## Threshold Sensitivity Analysis (Logistic Regression)

Brier score is the primary metric for this project since it evaluates probability quality directly and doesn't depend on any cutoff. But it's also worth checking how the model behaves as a binary classifier, since that requires picking a threshold to convert predicted probabilities into a podium or no podium call.

Tested three thresholds on the held out test set:

| Threshold | Precision (Podium) | Recall (Podium) | F1 (Podium) |
|---|---|---|---|
| 0.50 | 0.51 | 0.61 | 0.56 |
| 0.30 | 0.51 | 0.83 | 0.63 |
| 0.25 | 0.48 | 0.83 | 0.61 |

The default 0.5 threshold is too conservative for a relatively rare event like a podium finish, which only happens in about 15% of rows. Lowering the threshold to 0.3 catches significantly more real podiums, recall goes from 0.61 to 0.83, while precision stays essentially flat. That's a genuinely better tradeoff all things considered.

Pushing further to 0.25 doesn't help. Recall stays the same at 0.83, but precision drops a bit further, from 0.51 to 0.48, so 0.3 looks like the better operating point of the two.

This threshold choice is itself a modeling decision. Different applications might reasonably prefer different tradeoffs between catching more true podiums and avoiding false alarms. Since Brier score doesn't depend on any threshold, it stays the honest primary metric for comparing models, but this analysis adds useful depth on how the model behaves in practice.

## Limitations

### Weather Features (Tier 2)

Per-race weather conditions sourced via two parallel pipelines sharing a unified schema (`data/processed/weather.parquet`):

- **Training data (2022–2024):** actual session weather pulled via FastF1's `weather_data`, collapsed to one row per race (`rain_flag`, `avg_track_temp`, `avg_air_temp`, `humidity`).
- **Live 2026 inference:** pre-race forecasts pulled from Open-Meteo (free, no API key), using a static circuit lat/lon lookup (`data/circuits_lat_lon.csv`). Forecasts are matched to each race's local start hour via FastF1's 2026 event schedule (`Session5`, confirmed accurate on both standard and sprint-format weekends).

Both sources are merged directly on `(year, round_number)`  there is no leakage-safe shift needed, since weather is exogenous per-race information rather than a stat derived from past outcomes.

**Known limitations**
- **Train/inference mismatch:** training rows use *actual* recorded weather; live rows use a *forecast*, since the true outcome isn't knowable before the race. This is a standard forecast-dependency limitation, not data leakage as weather is knowable before lights out either way, just imperfectly at inference time. A `source` column (`'actual'` vs `'forecast'`) is retained in the raw weather table for auditing, but is not used as a model feature.
- **`avg_track_temp` is NaN for all forecast rows.** Open-Meteo does not forecast asphalt/track temperature (a function of solar load, not just air weather), only air temperature. Rather than approximate it with a proxy that could silently mislead the model, this field is left honestly missing for live predictions. `avg_air_temp` is the reliable live signal for this feature.
- **Career DNF rate**: uses extended 2018–2024 table for career-spanning depth; however this is a documented gap 2024 R3 (Australian GP) missing Ollie Bearman's row (one-off substitute, not captured in FastF1/Jolpica for that session).