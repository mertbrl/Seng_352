from __future__ import annotations

from pathlib import Path
from typing import Any
from datetime import date, datetime, timedelta

import joblib
import pandas as pd
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.preprocessing import (
    apply_rainfall_cap,
    apply_rare_weather_grouping,
    apply_scale_numeric,
    engineer_features,
    one_hot_encode_weather,
    separate_features_target,
)

DEFAULT_LATITUDE = 44.9537
DEFAULT_LONGITUDE = -93.09


def weather_code_to_main(weather_code: int | float | None) -> str:
    """Map Open-Meteo weather codes to dataset-style weather categories."""
    if weather_code is None or pd.isna(weather_code):
        return "Clear"

    code = int(weather_code)
    if code == 0:
        return "Clear"
    if code in {1, 2, 3, 45, 48}:
        return "Clouds"
    if code in {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99}:
        return "Rain"
    if code in {71, 73, 75, 77, 85, 86}:
        return "Snow"
    return "Mist"


def holiday_label(target_date: date) -> str:
    """Federal holiday flag; enough because the model only uses holiday vs none."""
    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(
        start=pd.Timestamp(target_date),
        end=pd.Timestamp(target_date),
    )
    return "Holiday" if len(holidays) else "None"


def fetch_weather_for_datetime(
    target_datetime: datetime,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    timeout: int = 10,
) -> dict[str, Any]:
    """Fetch hourly weather from Open-Meteo for app inference."""
    hourly_weather = fetch_hourly_weather_for_date(
        target_datetime,
        latitude=latitude,
        longitude=longitude,
        timeout=timeout,
    )
    target_hour = target_datetime.replace(minute=0, second=0, microsecond=0)
    if target_hour in hourly_weather:
        return hourly_weather[target_hour]

    nearest_hour = min(hourly_weather, key=lambda hour: abs(hour - target_hour))
    return hourly_weather[nearest_hour]


def fetch_hourly_weather_for_date(
    target_datetime: datetime,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    timeout: int = 10,
) -> dict[datetime, dict[str, Any]]:
    """Fetch one day of hourly weather for app inference and charts."""
    today = datetime.now().date()
    target_date = target_datetime.date()

    if target_date < today:
        url = "https://archive-api.open-meteo.com/v1/archive"
    elif target_date <= today + timedelta(days=16):
        url = "https://api.open-meteo.com/v1/forecast"
    else:
        raise ValueError(
            "Selected date is outside the weather API forecast window. "
            "Using demo fallback weather values instead."
        )

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
        "hourly": "temperature_2m,rain,snowfall,cloud_cover,weather_code",
        "timezone": "America/Chicago",
    }
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    hourly = response.json()["hourly"]

    weather_df = pd.DataFrame(hourly)
    weather_df["time"] = pd.to_datetime(weather_df["time"])

    hourly_weather = {}
    for _, row in weather_df.iterrows():
        weather_main = weather_code_to_main(row.get("weather_code"))
        hour = row["time"].to_pydatetime().replace(minute=0, second=0, microsecond=0)
        hourly_weather[hour] = {
            "temp_c": float(row.get("temperature_2m", 15.0)),
            "rain_1h": float(row.get("rain", 0.0) or 0.0),
            "snow_1h": float(row.get("snowfall", 0.0) or 0.0),
            "clouds_all": int(round(float(row.get("cloud_cover", 40.0) or 0.0))),
            "weather_main": weather_main,
            "weather_description": weather_main.lower(),
            "source": "Open-Meteo",
        }
    return hourly_weather


def build_prediction_row(
    target_datetime: datetime,
    weather: dict[str, Any],
) -> dict[str, Any]:
    """Create one model-ready input row for dashboard prediction."""
    temp_c = float(weather.get("temp_c", 15.0))
    weather_main = weather.get("weather_main", "Clear")
    return {
        "date_time": target_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "holiday": holiday_label(target_datetime.date()),
        "temp": temp_c + 273.15,
        "rain_1h": float(weather.get("rain_1h", 0.0)),
        "snow_1h": float(weather.get("snow_1h", 0.0)),
        "clouds_all": int(weather.get("clouds_all", 40)),
        "weather_main": weather_main,
        "weather_description": weather.get("weather_description", str(weather_main).lower()),
    }


def load_model_and_artifacts(
    model_path: str | Path,
    artifacts_path: str | Path,
) -> tuple[Any, dict[str, Any]]:
    model = joblib.load(model_path)
    artifacts = joblib.load(artifacts_path)
    return model, artifacts


def prepare_inference_features(input_data: dict[str, Any] | pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    """Apply the same feature logic used during training."""
    df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data.copy()
    df["date_time"] = pd.to_datetime(df["date_time"])

    if "traffic_volume" not in df.columns:
        df["traffic_volume"] = 0

    df = apply_rainfall_cap(df, artifacts["rain_cap"])
    df = engineer_features(df)
    df = apply_rare_weather_grouping(df, artifacts["common_weather_categories"])
    df, _ = one_hot_encode_weather(df, dummy_columns=artifacts["dummy_columns"])

    X, _ = separate_features_target(df)
    X = X.reindex(columns=artifacts["feature_columns"], fill_value=0)

    if artifacts.get("scaler") is not None:
        X = apply_scale_numeric(X, artifacts["scaler"], artifacts["scale_columns"])

    return X[artifacts["selected_features"]]


def predict_traffic_volume(
    input_data: dict[str, Any] | pd.DataFrame,
    model_path: str | Path = "models/final_model.joblib",
    artifacts_path: str | Path = "models/preprocessing_artifacts.joblib",
) -> pd.Series:
    model, artifacts = load_model_and_artifacts(model_path, artifacts_path)
    X = prepare_inference_features(input_data, artifacts)
    return pd.Series(model.predict(X), name="predicted_traffic_volume")
