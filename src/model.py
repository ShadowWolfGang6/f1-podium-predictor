from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

def prepare_model_features_onehot_team(df, label_encoders=None, team_columns=None):
    """
    Variant of prepare_model_features for logistic regression, one hot
    encoding TeamId instead of label encoding it. Linear models treat a
    label encoded categorical as if it has a meaningful numeric order,
    which is not true for team identity, one hot encoding avoids that
    false assumption. DriverId, PU_Group, and Location stay label encoded
    since one hot encoding them would add many more columns for a small
    dataset.

    team_columns: the exact set of one hot TeamId columns from train,
    passed in when encoding val or test so both splits end up with the
    same columns in the same order, even if a team is missing from a
    given split. Pass None when encoding train for the first time.
    """
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()
    df["Podium"] = (df["Position"] <= 3).astype(int)

    df = df.drop(columns=LEAKAGE_COLS + IDENTIFIER_COLS + ["TeamName"])

    df["TeamId"] = df["TeamId"].replace(TEAM_REBRAND_MAP)

    label_cols = ["DriverId", "PU_Group", "Location"]

    fit_new = label_encoders is None
    if fit_new:
        label_encoders = {}

    for col in label_cols:
        if fit_new:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            df[col] = df[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    team_dummies = pd.get_dummies(df["TeamId"], prefix="Team")
    df = df.drop(columns=["TeamId"])

    if team_columns is None:
        team_columns = team_dummies.columns.tolist()
    else:
        team_dummies = team_dummies.reindex(columns=team_columns, fill_value=0)

    df = pd.concat([df, team_dummies.astype(int)], axis=1)

    df["is_cold_start"] = df["is_cold_start"].astype(int)
    df["rain_flag"] = df["rain_flag"].astype(int)

    return df, label_encoders, team_columns