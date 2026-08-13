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

## Gradient Boosted Model: XGBoost

Built an XGBoost classifier on top of the same feature table, using label encoded categoricals directly rather than one hot encoding, since tree based models split on thresholds and do not assume a false numeric order for identity features like TeamId. This was the direct payoff of the earlier one hot versus label encoding investigation done for logistic regression.

**Hyperparameter search:** ran two rounds of grid search on the validation set, no random k fold cross validation, since the project's time based split already respects chronological order and standard cross validation would reintroduce leakage. The first search covered a modest grid across max_depth, n_estimators, and learning_rate. A second, refined search widened the range of n_estimators and narrowed in on the learning rates that performed best in the first round. Both searches converged on the same winning configuration, max_depth=4, n_estimators=50, learning_rate=0.1, a good sign the result is stable and not up to pure luck.

**Results (Brier score, lower is better):**

| | Validation | Test |
|---|---|---|
| Grid position baseline | 0.0822 | 0.1250 |
| Logistic regression | 0.0644 | 0.0956 |
| XGBoost (tuned) | 0.0632 | 0.0886 |

XGBoost beat logistic regression by roughly 7 percent and the grid baseline by roughly 29 percent on the held out test set. The improvement held in the same direction from validation to test across every model, none of them reversed or collapsed on unseen data, a good sign the whole pipeline generalizes rather than overfitting to one particular slice.

**Feature importance:** the top features from XGBoost closely match the top coefficients from logistic regression, GridPosition dominates by a wide margin in both, followed by Drivers_Standings, RollingFinish, and DiscountedCareerPosition. This agreement between two structurally different models is a strong, independent validation that the feature engineering captured real signal rather than something specific to one algorithm.

## 2025 Season Backtest

Now, using the 2025 season's results, both models were retrained on all prior history through 2024 (2022-2024 combined, rather than the original 2022-2023 only split) and used to predict every 2025 race. This mirrors how the models will actually be used for live 2026 predictions (2026 Dutch GP onward) relying on every prior season available at prediction time. Hyperparameters were reused exactly as tuned earlier, not retuned against 2025, since doing so would leak information into what is meant to mirror a clean holdout test.

**Results (Brier score, lower is better):**

| | 2024 Test Set | 2025 Full Season |
|---|---|---|
| Logistic regression | 0.0956 | 0.0685 |
| XGBoost | 0.0886 | 0.0678 |

Both models improved meaningfully on 2025 compared to the original 2024 test, likely because they were trained on more history, 2022-2024 combined versus 2022-2023 only, exactly the benefit you would expect and rely on for real predictions going forward.

**On Brier score alone, the two models are essentially tied**, 0.0678 versus 0.0685, a difference small enough to be within noise. But a second, more intuitive metric tells a different story. Counting how often each model's top three predicted probabilities included an actual podium finisher, summed across all 23 scored rounds:

| | Correct podium picks (of 69 possible) |
|---|---|
| Logistic regression | 50 (72.5%) |
| XGBoost | 46 (66.7%) |

Logistic regression noticeably outperformed XGBoost on this specific measure. Reviewing the round by round predictions, XGBoost struggled more through a mid to late season stretch that included several genuinely low probability surprise results, Hulkenberg's podium from P19 at round 12 and Antonelli's podium from P17 at round 22 among them, results no pre-race model would reasonably be expected to call. I believe this is a genuine nuanced finding rather than a simple win for one model: XGBoost and logistic regression are close on probability quality overall, but logistic regression's simpler, lower variance nature appears to have produced more stable top three picks across this particular season.

Round 7 (Imola) is excluded from the backtest, consistent with the project's existing handling of off calendar circuits.

**Data recovery note:** during 2025 ingestion, three separate get_all_* functions (results, weather, pit stops) overwrote their combined output files instead of merging with existing 2022-2024 data, a real bug caught and fixed during this session. All three were recovered cleanly from existing per year checkpoint files with no data loss and no re-ingestion needed. A separate issue was also found and fixed during this process, FastF1 labeled the Miami circuit as "Miami Gardens" in 2025 event data, a naming drift from prior seasons that messed up the tire degradation and archetype merge for that round. Both fixes are documented in the codebase and applied going forward.

## Limitations

**Calibration**
Per the original project plan, checked whether stated probabilities matched real world outcomes, does a model saying "70 percent chance of podium" actually correspond to a podium roughly 70 percent of the time.

**Initial finding:** built a calibration curve on the 2024 test set for both models. Both were reasonably well calibrated at low predicted probabilities, but noticeably overconfident in the middle to high range. A predicted probability around 80 percent corresponded to an actual podium rate closer to 50 percent in the data.

**Attempted fix:** applied scikit-learn's CalibratedClassifierCV (sigmoid method, 5 fold cross validation) to both models, the standard approach named directly in the original project plan.

**Result:** calibration made both models measurably worse on Brier score, not better.

| | Uncalibrated | Calibrated |
|---|---|---|
| Logistic regression | 0.0956 | 0.1067 |
| XGBoost | 0.0886 | 0.0930 |

The calibrated models compressed nearly all predicted probabilities toward the middle of the range, the highest calibrated prediction across the entire test set was only about 0.53, compared to several uncalibrated predictions above 0.80 that were often correct, most notably for dominant drivers like Verstappen. This is a direct consequence of dataset size. CalibratedClassifierCV's cross validation splits an already modest 838 row training set into five folds of roughly 170 rows each, too little data for the calibration mapping itself to learn a stable correction without over correcting toward caution.

**Decision:** calibration was tested, measured, and rejected for this iteration of the project. The original, uncalibrated models remain the ones used for live 2026 predictions. This is a a real limitation rather than my oversight, as the original overconfidence issue at high probabilities is already documented, but the standard fix for it costs more than it gains given the current training set size.

### Weather Features (Tier 2)

Per-race weather conditions sourced via two parallel pipelines sharing a unified schema (`data/processed/weather.parquet`):

- **Training data (2022–2024):** actual session weather pulled via FastF1's `weather_data`, collapsed to one row per race (`rain_flag`, `avg_track_temp`, `avg_air_temp`, `humidity`).
- **Live 2026 inference:** pre-race forecasts pulled from Open-Meteo (free, no API key), using a static circuit lat/lon lookup (`data/circuits_lat_lon.csv`). Forecasts are matched to each race's local start hour via FastF1's 2026 event schedule (`Session5`, confirmed accurate on both standard and sprint-format weekends).

Both sources are merged directly on `(year, round_number)`  there is no leakage-safe shift needed, since weather is exogenous per-race information rather than a stat derived from past outcomes.

**Known limitations**
- **Train/inference mismatch:** training rows use *actual* recorded weather; live rows use a *forecast*, since the true outcome isn't knowable before the race. This is a standard forecast-dependency limitation, not data leakage as weather is knowable before lights out either way, just imperfectly at inference time. A `source` column (`'actual'` vs `'forecast'`) is retained in the raw weather table for auditing, but is not used as a model feature.
- **`avg_track_temp` is NaN for all forecast rows.** Open-Meteo does not forecast asphalt/track temperature (a function of solar load, not just air weather), only air temperature. Rather than approximate it with a proxy that could silently mislead the model, this field is left missing for live predictions. `avg_air_temp` is the reliable live signal for this feature.
- **Career DNF rate**: uses extended 2018–2024 table for career-spanning depth; however this is a documented gap 2024 R3 (Australian GP) missing Ollie Bearman's row (one-off substitute, not captured in FastF1/Jolpica for that session).