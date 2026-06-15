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
*To be updated as model is built and validated.*

## Limitations
*To be updated honestly after backtesting.*