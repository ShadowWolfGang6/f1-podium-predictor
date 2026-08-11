from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from itertools import product
import pandas as pd

METADATA_COLS = ["year", "round_number"]
TARGET_COL = "Podium"


def grid_position_baseline_brier(df):
    """
    Baseline per project spec: grid position equals finishing position.
    Predicts podium (1) if GridPosition is 1, 2, or 3, else predicts no
    podium (0). This is the baseline the actual model needs to beat,
    """
    y_true = df["Podium"]
    y_pred = (df["GridPosition"] <= 3).astype(int)

    score = brier_score_loss(y_true, y_pred)
    return score

def get_feature_columns(df):
    """
    Returns the list of columns to use as model features, excluding the
    target and raw temporal metadata (year, round_number). year and
    round_number are excluded because they are not meaningful predictive
    signals on their own, the engineered era/round features
    (EraStartYear, EraRaceNumber, GroundEffectEra, ActiveAeroEra) already
    capture the relevant regulation era context.
    """
    exclude = METADATA_COLS + [TARGET_COL]
    return [col for col in df.columns if col not in exclude]

def fit_logistic_baseline(train, val):
    """
    Fits a logistic regression baseline on train, scaled via StandardScaler
    fit on train only (applied to val without refitting, avoiding leakage).
    Returns the fitted model, the scaler, and the val Brier score.
    """
    feature_cols = get_feature_columns(train)

    X_train = train[feature_cols]
    y_train = train[TARGET_COL]

    X_val = val[feature_cols]
    y_val = val[TARGET_COL]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    val_probs = model.predict_proba(X_val_scaled)[:, 1]
    val_score = brier_score_loss(y_val, val_probs)

    return model, scaler, val_score, feature_cols

def fit_xgboost_baseline(train, val, max_depth=4, n_estimators=100, learning_rate=0.1):
    """
    Fits an XGBoost classifier on train using label encoded categoricals,
    no scaling needed since tree models split on thresholds regardless of
    feature units. Returns the fitted model, val Brier score, and feature
    columns used.

    Starting with modest hyperparameters (max_depth=4, n_estimators=100,
    learning_rate=0.1) as a first pass, not yet tuned.
    """
    feature_cols = get_feature_columns(train)

    X_train = train[feature_cols]
    y_train = train[TARGET_COL]

    X_val = val[feature_cols]
    y_val = val[TARGET_COL]

    model = xgb.XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_score = brier_score_loss(y_val, val_probs)

    return model, val_score, feature_cols

from itertools import product

def grid_search_xgboost(train, val, param_grid=None):
    """
    Simple grid search over XGBoost hyperparameters, evaluated on val
    Brier score. Not using cross-validation since the project's time
    based split (train on 2022-2023, val on 2024 rounds 1-12) already
    respects chronological order, standard k-fold cross-validation
    would reintroduce the leakage this project has been careful to avoid.

    Returns the best model, best params, and a results table for all
    combinations tried, so the full search is documented.
    """
    if param_grid is None:
        param_grid = {
            "max_depth": [2, 3, 4],
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.01, 0.05, 0.1],
        }

    feature_cols = get_feature_columns(train)
    X_train = train[feature_cols]
    y_train = train[TARGET_COL]
    X_val = val[feature_cols]
    y_val = val[TARGET_COL]

    results = []
    best_score = float("inf")
    best_model = None
    best_params = None

    keys = list(param_grid.keys())
    for values in product(*param_grid.values()):
        params = dict(zip(keys, values))

        model = xgb.XGBClassifier(
            **params,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train, y_train)

        val_probs = model.predict_proba(X_val)[:, 1]
        score = brier_score_loss(y_val, val_probs)

        results.append({**params, "val_brier": score})

        if score < best_score:
            best_score = score
            best_model = model
            best_params = params

    results_df = pd.DataFrame(results).sort_values("val_brier")

    return best_model, best_params, best_score, results_df, feature_cols