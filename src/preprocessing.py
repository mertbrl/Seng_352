from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import RobustScaler


TARGET = "traffic_volume"
DATE_COL = "date_time"
RAIN_COL = "rain_1h"
SNOW_COL = "snow_1h"
TEMP_COL = "temp"
CLOUD_COL = "clouds_all"
WEATHER_COL = "weather_main"
GROUPED_WEATHER_COL = "weather_main_grouped"

RAW_COLUMNS_TO_DROP = [
    TARGET,
    DATE_COL,
    "holiday",
    WEATHER_COL,
    "weather_description",
]

NUMERIC_SCALE_COLUMNS = [
    TEMP_COL,
    RAIN_COL,
    SNOW_COL,
    CLOUD_COL,
    "hour",
    "day_of_week",
    "month",
    "year",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "temp_c",
    "temp_c2",
    "rain_log",
    "snow_log",
    "clouds_all_pct",
]

BINARY_FEATURE_COLUMNS = [
    "is_weekend",
    "is_rush_hour",
    "is_night",
    "is_holiday",
    "is_raining",
    "is_snowing",
]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Read data, parse dates, and sort chronologically."""
    df = pd.read_csv(csv_path, parse_dates=[DATE_COL])
    return df.sort_values(DATE_COL).reset_index(drop=True)


def global_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Remove only physically invalid records before splitting."""
    cleaned = df.copy()
    cleaned = cleaned[cleaned[TEMP_COL] != 0].reset_index(drop=True)
    return cleaned


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split to avoid leakage from future data."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    n_rows = len(df)
    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * val_ratio)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def fit_rainfall_cap(train_df: pd.DataFrame, percentile: float = 0.995) -> float:
    """Learn rainfall cap from training data only."""
    return float(train_df[RAIN_COL].quantile(percentile))


def apply_rainfall_cap(df: pd.DataFrame, rain_cap: float) -> pd.DataFrame:
    capped = df.copy()
    capped[RAIN_COL] = capped[RAIN_COL].clip(upper=rain_cap)
    return capped


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["hour"] = featured[DATE_COL].dt.hour
    featured["day_of_week"] = featured[DATE_COL].dt.dayofweek
    featured["month"] = featured[DATE_COL].dt.month
    featured["year"] = featured[DATE_COL].dt.year

    featured["is_weekend"] = (featured["day_of_week"] >= 5).astype(int)
    featured["is_rush_hour"] = featured["hour"].isin([7, 8, 16, 17, 18]).astype(int)  # rush-hour indicator
    featured["is_night"] = ((featured["hour"] <= 5) | (featured["hour"] >= 22)).astype(int)

    holiday_text = featured["holiday"].fillna("None").astype(str).str.lower()
    featured["is_holiday"] = (holiday_text != "none").astype(int)  # holiday flag

    featured["hour_sin"] = np.sin(2 * np.pi * featured["hour"] / 24)  # cyclical encoding for hour
    featured["hour_cos"] = np.cos(2 * np.pi * featured["hour"] / 24)
    featured["dow_sin"] = np.sin(2 * np.pi * featured["day_of_week"] / 7)  # cyclical encoding for weekday
    featured["dow_cos"] = np.cos(2 * np.pi * featured["day_of_week"] / 7)
    featured["month_sin"] = np.sin(2 * np.pi * featured["month"] / 12)  # cyclical encoding for month
    featured["month_cos"] = np.cos(2 * np.pi * featured["month"] / 12)
    return featured


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["temp_c"] = featured[TEMP_COL] - 273.15
    featured["temp_c2"] = featured["temp_c"] ** 2
    featured["rain_log"] = np.log1p(featured[RAIN_COL])  # log after rainfall capping
    featured["snow_log"] = np.log1p(featured[SNOW_COL])
    featured["is_raining"] = (featured[RAIN_COL] > 0).astype(int)
    featured["is_snowing"] = (featured[SNOW_COL] > 0).astype(int)
    featured["clouds_all_pct"] = featured[CLOUD_COL] / 100.0
    return featured


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal and weather features."""
    return add_weather_features(add_time_features(df))


def fit_rare_weather_categories(train_df: pd.DataFrame, min_freq: float = 0.01) -> list[str]:
    """Learn non-rare weather categories from training data only."""
    frequencies = train_df[WEATHER_COL].value_counts(normalize=True)
    return frequencies[frequencies >= min_freq].index.tolist()


def apply_rare_weather_grouping(
    df: pd.DataFrame,
    common_categories: list[str],
) -> pd.DataFrame:
    grouped = df.copy()
    grouped[GROUPED_WEATHER_COL] = grouped[WEATHER_COL].where(
        grouped[WEATHER_COL].isin(common_categories),
        "Other",
    )
    return grouped


def one_hot_encode_weather(
    df: pd.DataFrame,
    dummy_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    encoded = pd.get_dummies(df, columns=[GROUPED_WEATHER_COL], prefix="weather_main", dtype=int)
    if dummy_columns is None:
        dummy_columns = [col for col in encoded.columns if col.startswith("weather_main_")]
    else:
        for col in dummy_columns:
            if col not in encoded.columns:
                encoded[col] = 0

    weather_cols_now = [col for col in encoded.columns if col.startswith("weather_main_")]
    keep_weather = set(dummy_columns)
    encoded = encoded.drop(columns=[col for col in weather_cols_now if col not in keep_weather])
    return encoded, dummy_columns


def separate_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = [col for col in RAW_COLUMNS_TO_DROP if col in df.columns]
    X = df.drop(columns=drop_cols)
    X = X.select_dtypes(include=[np.number, "bool"]).copy()
    y = df[TARGET].copy()
    return X, y


def filter_training_outliers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    contamination: float = 0.01,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, IsolationForest, pd.Series]:
    """Isolation Forest on training set only."""
    detector = IsolationForest(contamination=contamination, random_state=random_state)
    inlier_mask = pd.Series(detector.fit_predict(X_train) == 1, index=X_train.index)
    return (
        X_train.loc[inlier_mask].reset_index(drop=True),
        y_train.loc[inlier_mask].reset_index(drop=True),
        detector,
        inlier_mask,
    )


def fit_scale_numeric(
    X_train: pd.DataFrame,
    scale_columns: list[str] | None = None,
) -> tuple[RobustScaler, list[str]]:
    columns = scale_columns or [
        col
        for col in NUMERIC_SCALE_COLUMNS
        if col in X_train.columns and col not in BINARY_FEATURE_COLUMNS
    ]
    scaler = RobustScaler()
    scaler.fit(X_train[columns])
    return scaler, columns


def apply_scale_numeric(
    X: pd.DataFrame,
    scaler: RobustScaler,
    scale_columns: list[str],
) -> pd.DataFrame:
    scaled = X.copy()
    scaled[scale_columns] = scaler.transform(scaled[scale_columns])
    return scaled


def select_features_with_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    mode: str = "top_k",
    top_k: int = 25,
    importance_threshold: float = 0.01,
    random_state: int = 42,
) -> tuple[list[str], pd.DataFrame]:
    """Feature selection using training-set Random Forest importance."""
    selector = RandomForestRegressor(
        n_estimators=150,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=5,
    )
    selector.fit(X_train, y_train)
    importances = pd.DataFrame(
        {"feature": X_train.columns, "importance": selector.feature_importances_}
    ).sort_values("importance", ascending=False)

    if mode == "top_k":
        selected = importances.head(min(top_k, len(importances)))["feature"].tolist()
    elif mode == "importance_threshold":
        selected = importances.loc[
            importances["importance"] >= importance_threshold,
            "feature",
        ].tolist()
    else:
        raise ValueError("Feature selection mode must be 'top_k' or 'importance_threshold'")

    if not selected:
        selected = importances.head(min(top_k, len(importances)))["feature"].tolist()
    return selected, importances


def prepare_train_val_test(
    df: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Full leakage-aware preprocessing pipeline."""
    seed = params.get("random_seed", 42)
    split_params = params.get("split", {})
    prep_params = params.get("preprocessing", {})
    fs_params = params.get("feature_selection", {})

    clean_df = global_cleaning(df)
    train_df, val_df, test_df = chronological_split(
        clean_df,
        train_ratio=split_params.get("train_ratio", 0.70),
        val_ratio=split_params.get("val_ratio", 0.15),
        test_ratio=split_params.get("test_ratio", 0.15),
    )

    rain_cap = fit_rainfall_cap(
        train_df,
        percentile=prep_params.get("rainfall_cap_percentile", 0.995),
    )
    train_df = engineer_features(apply_rainfall_cap(train_df, rain_cap))
    val_df = engineer_features(apply_rainfall_cap(val_df, rain_cap))
    test_df = engineer_features(apply_rainfall_cap(test_df, rain_cap))

    common_weather = fit_rare_weather_categories(
        train_df,
        min_freq=prep_params.get("rare_category_min_freq", 0.01),
    )
    train_df = apply_rare_weather_grouping(train_df, common_weather)
    val_df = apply_rare_weather_grouping(val_df, common_weather)
    test_df = apply_rare_weather_grouping(test_df, common_weather)

    train_df, dummy_columns = one_hot_encode_weather(train_df)
    val_df, _ = one_hot_encode_weather(val_df, dummy_columns=dummy_columns)
    test_df, _ = one_hot_encode_weather(test_df, dummy_columns=dummy_columns)

    X_train, y_train = separate_features_target(train_df)
    X_val, y_val = separate_features_target(val_df)
    X_test, y_test = separate_features_target(test_df)

    feature_columns = X_train.columns.tolist()
    X_val = X_val.reindex(columns=feature_columns, fill_value=0)
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    outlier_detector = None
    outlier_mask = None
    if prep_params.get("use_isolation_forest", True):
        X_train, y_train, outlier_detector, outlier_mask = filter_training_outliers(
            X_train,
            y_train,
            contamination=prep_params.get("isolation_forest_contamination", 0.01),
            random_state=seed,
        )

    scaler = None
    scale_columns: list[str] = []
    if prep_params.get("scale_numeric", True):
        scaler, scale_columns = fit_scale_numeric(X_train)
        X_train = apply_scale_numeric(X_train, scaler, scale_columns)
        X_val = apply_scale_numeric(X_val, scaler, scale_columns)
        X_test = apply_scale_numeric(X_test, scaler, scale_columns)

    selected_features = feature_columns
    feature_importance = pd.DataFrame()
    if fs_params.get("enabled", True):
        selected_features, feature_importance = select_features_with_random_forest(
            X_train,
            y_train,
            mode=fs_params.get("mode", "top_k"),
            top_k=fs_params.get("top_k", 25),
            importance_threshold=fs_params.get("importance_threshold", 0.01),
            random_state=seed,
        )
        X_train = X_train[selected_features]
        X_val = X_val[selected_features]
        X_test = X_test[selected_features]

    artifacts = {
        "rain_cap": rain_cap,
        "common_weather_categories": common_weather,
        "dummy_columns": dummy_columns,
        "feature_columns": feature_columns,
        "selected_features": selected_features,
        "scaler": scaler,
        "scale_columns": scale_columns,
        "outlier_detector": outlier_detector,
        "feature_importance": feature_importance,
        "params": params,
    }

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "outlier_mask": outlier_mask,
        "artifacts": artifacts,
    }


def save_preprocessing_artifacts(artifacts: dict[str, Any], output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, output_path)


def load_preprocessing_artifacts(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)
