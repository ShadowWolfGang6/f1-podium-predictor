import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss

from src.features import build_feature_table
from src.impute import (
    drop_unusable_rows,
    prepare_model_features,
    prepare_model_features_onehot_team,
    COLD_START_COLS,
)
from src.model import get_feature_columns, TARGET_COL


def compute_cold_start_means(train, cold_start_cols=COLD_START_COLS):
    return train[cold_start_cols].mean()


def apply_cold_start_imputation(df, train_means, cold_start_cols=COLD_START_COLS):
    """
    Applies cold start imputation using means already computed from a
    training set, rather than fitting new means. This keeps the 2025
    backtest honest, the imputation values come only from 2022 to 2024,
    nothing from 2025 itself leaks into how its own gaps get filled.
    """
    df = df.copy()
    df["is_cold_start"] = df[cold_start_cols].isna().any(axis=1)
    df[cold_start_cols] = df[cold_start_cols].fillna(train_means)
    return df


def run_2025_backtest():
    """
    Trains both models on all available history through 2024, then
    predicts on the full 2025 season. This mirrors how the models would
    actually be used for live 2026 predictions, using every prior season
    as training data rather than holding some of it back.

    Hyperparameters are reused as already tuned against the original val
    split, not retuned here, since tuning against 2025 would leak
    information into what is meant to be a clean holdout test.
    """
    df = build_feature_table()
    df = drop_unusable_rows(df)

    train_full = df[df["year"] <= 2024].copy()
    season_2025 = df[df["year"] == 2025].copy()
    season_2025["Podium"] = (season_2025["Position"] <= 3).astype(int)

    train_means = compute_cold_start_means(train_full)
    train_full = apply_cold_start_imputation(train_full, train_means)
    season_2025_imputed = apply_cold_start_imputation(season_2025, train_means)

    # logistic regression, one hot team encoding
    train_lr, lr_encoders, team_cols = prepare_model_features_onehot_team(train_full)
    season_2025_lr, _, _ = prepare_model_features_onehot_team(
        season_2025_imputed, label_encoders=lr_encoders, team_columns=team_cols
    )

    lr_feature_cols = get_feature_columns(train_lr)
    X_train_lr = train_lr[lr_feature_cols]
    y_train_lr = train_lr[TARGET_COL]
    X_2025_lr = season_2025_lr[lr_feature_cols]

    scaler = StandardScaler()
    X_train_lr_scaled = scaler.fit_transform(X_train_lr)
    X_2025_lr_scaled = scaler.transform(X_2025_lr)

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_lr_scaled, y_train_lr)
    lr_probs = lr_model.predict_proba(X_2025_lr_scaled)[:, 1]

    # xgboost, label encoded categoricals
    train_xgb, xgb_encoders = prepare_model_features(train_full)
    season_2025_xgb, _ = prepare_model_features(season_2025_imputed, label_encoders=xgb_encoders)

    xgb_feature_cols = get_feature_columns(train_xgb)
    X_train_xgb = train_xgb[xgb_feature_cols]
    y_train_xgb = train_xgb[TARGET_COL]
    X_2025_xgb = season_2025_xgb[xgb_feature_cols]

    xgb_model = xgb.XGBClassifier(
        max_depth=4,
        n_estimators=50,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
    )
    xgb_model.fit(X_train_xgb, y_train_xgb)
    xgb_probs = xgb_model.predict_proba(X_2025_xgb)[:, 1]

    results = season_2025[["year", "round_number", "DriverId", "GridPosition", "Position", "Podium"]].copy()
    results["LR_PodiumProb"] = lr_probs
    results["XGB_PodiumProb"] = xgb_probs

    results["LR_PredictedRank"] = results.groupby(["year", "round_number"])["LR_PodiumProb"].rank(
        ascending=False, method="first"
    )
    results["XGB_PredictedRank"] = results.groupby(["year", "round_number"])["XGB_PodiumProb"].rank(
        ascending=False, method="first"
    )

    return results, lr_model, xgb_model


def score_2025_backtest(results):
    lr_brier = brier_score_loss(results["Podium"], results["LR_PodiumProb"])
    xgb_brier = brier_score_loss(results["Podium"], results["XGB_PodiumProb"])
    return lr_brier, xgb_brier


def print_round_comparison(results, round_number, model="XGB"):
    """
    Prints predicted probability, predicted rank, actual finishing
    position, and actual podium status for every driver in a given round,
    sorted by the chosen model's predicted rank.
    """
    r = results[results["round_number"] == round_number].copy()
    prob_col = f"{model}_PodiumProb"
    rank_col = f"{model}_PredictedRank"
    r = r.sort_values(rank_col)

    print(f"--- Round {round_number} ({model}) ---")
    print(f"{'Driver':16} {'Grid':5} {'PredProb':9} {'PredRank':9} {'ActualPos':10} {'Podium':6}")
    for _, row in r.iterrows():
        podium_flag = "YES" if row["Podium"] == 1 else ""
        print(
            f"{row['DriverId']:16} {row['GridPosition']:<5.0f} "
            f"{row['LR_PodiumProb' if model == 'LR' else 'XGB_PodiumProb']:<9.3f} "
            f"{int(row[rank_col]):<9} {row['Position']:<10.0f} {podium_flag:6}"
        )

def print_round_summary(results, round_number):
    """
    Prints both models' predicted top 3 (by podium probability) against
    the actual podium finishers for a single round, plus each model's
    predicted probability for those drivers.
    """
    r = results[results["round_number"] == round_number].copy()

    actual_podium = r[r["Podium"] == 1].sort_values("Position")
    lr_predicted = r.sort_values("LR_PodiumProb", ascending=False).head(3)
    xgb_predicted = r.sort_values("XGB_PodiumProb", ascending=False).head(3)

    print(f"===== Round {round_number} =====")
    print("Actual podium:")
    for _, row in actual_podium.iterrows():
        print(f"  P{int(row['Position'])}  {row['DriverId']} (started P{int(row['GridPosition'])})")

    print("Logistic regression predicted podium:")
    for _, row in lr_predicted.iterrows():
        hit = "HIT" if row["Podium"] == 1 else "miss"
        print(f"  {row['DriverId']:16} prob={row['LR_PodiumProb']:.3f}  actual finish P{int(row['Position'])}  [{hit}]")

    print("XGBoost predicted podium:")
    for _, row in xgb_predicted.iterrows():
        hit = "HIT" if row["Podium"] == 1 else "miss"
        print(f"  {row['DriverId']:16} prob={row['XGB_PodiumProb']:.3f}  actual finish P{int(row['Position'])}  [{hit}]")

    print()


def print_all_rounds(results):
    for round_number in sorted(results["round_number"].unique()):
        print_round_summary(results, round_number)