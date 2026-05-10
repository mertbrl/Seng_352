from datetime import date as date_type
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.inference import (
    build_prediction_row,
    fetch_hourly_weather_for_date,
    load_model_and_artifacts,
    prepare_inference_features,
)


st.set_page_config(
    page_title="I-94 Traffic Volume Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

model_path = PROJECT_ROOT / "models" / "final_model.joblib"
artifacts_path = PROJECT_ROOT / "models" / "preprocessing_artifacts.joblib"
raw_data_path = PROJECT_ROOT / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv"


WEATHER_THEME = {
    "Clear": {
        "icon": "SUN",
        "class": "sunny",
        "label": "Clear sky conditions",
        "gradient": "linear-gradient(135deg, #101824 0%, #25344a 48%, #f4b860 150%)",
    },
    "Clouds": {
        "icon": "CLOUD",
        "class": "cloudy",
        "label": "Cloud cover expected",
        "gradient": "linear-gradient(135deg, #101824 0%, #233044 55%, #6f7f91 150%)",
    },
    "Rain": {
        "icon": "RAIN",
        "class": "rainy",
        "label": "Rain may affect driving",
        "gradient": "linear-gradient(135deg, #08111f 0%, #123050 55%, #1f6f8b 150%)",
    },
    "Snow": {
        "icon": "SNOW",
        "class": "snowy",
        "label": "Snowy conditions possible",
        "gradient": "linear-gradient(135deg, #111827 0%, #2d4863 55%, #d7eef9 150%)",
    },
    "Mist": {
        "icon": "FOG",
        "class": "foggy",
        "label": "Reduced visibility possible",
        "gradient": "linear-gradient(135deg, #111827 0%, #354052 55%, #a8b0bd 150%)",
    },
    "Other": {
        "icon": "METEO",
        "class": "cloudy",
        "label": "Mixed weather conditions",
        "gradient": "linear-gradient(135deg, #101824 0%, #243045 55%, #60758d 150%)",
    },
}

SEVERITY_STYLE = {
    "Low": {
        "color": "#2dd4bf",
        "soft": "rgba(45, 212, 191, 0.16)",
        "label": "Light traffic expected",
        "note": "Roads should be relatively open.",
    },
    "Moderate": {
        "color": "#f59e0b",
        "soft": "rgba(245, 158, 11, 0.18)",
        "label": "Moderate traffic expected",
        "note": "Some congestion is possible.",
    },
    "High": {
        "color": "#ef4444",
        "soft": "rgba(239, 68, 68, 0.18)",
        "label": "Heavy traffic expected",
        "note": "Plan for slower travel.",
    },
}


@st.cache_data(show_spinner=False)
def load_traffic_thresholds() -> dict[str, float]:
    """Quantile thresholds from the chronological training target distribution."""
    df = pd.read_csv(raw_data_path, parse_dates=["date_time"])
    df = df.sort_values("date_time")
    df = df[df["temp"] != 0]
    train_end = int(len(df) * 0.70)
    y_train = df.iloc[:train_end]["traffic_volume"]
    return {
        "low_max": float(y_train.quantile(0.33)),
        "moderate_max": float(y_train.quantile(0.66)),
        "max_reference": float(y_train.quantile(0.95)),
    }


@st.cache_resource(show_spinner=False)
def load_prediction_assets():
    return load_model_and_artifacts(model_path, artifacts_path)


def classify_traffic(prediction: float, thresholds: dict[str, float]) -> str:
    if prediction <= thresholds["low_max"]:
        return "Low"
    if prediction <= thresholds["moderate_max"]:
        return "Moderate"
    return "High"


def severity_progress(prediction: float, thresholds: dict[str, float]) -> float:
    reference = max(thresholds["max_reference"], 1.0)
    return min(max(prediction / reference, 0.0), 1.0)


def weather_theme(weather_main: str) -> dict[str, str]:
    return WEATHER_THEME.get(weather_main, WEATHER_THEME["Other"])


def weather_effect_markup(weather_class: str) -> str:
    if weather_class == "sunny":
        return '<div class="sun-disc"></div><div class="sun-ray ray-one"></div><div class="sun-ray ray-two"></div>'
    if weather_class == "cloudy":
        return '<div class="cloud cloud-one"></div><div class="cloud cloud-two"></div>'
    if weather_class == "rainy":
        return '<div class="rain-field"><span></span><span></span><span></span><span></span><span></span><span></span></div>'
    if weather_class == "snowy":
        return '<div class="snow-field"><span></span><span></span><span></span><span></span><span></span><span></span></div>'
    if weather_class == "foggy":
        return '<div class="fog-line fog-one"></div><div class="fog-line fog-two"></div><div class="fog-line fog-three"></div>'
    return '<div class="cloud cloud-one"></div>'


def fallback_weather_values(temp_c, rain_1h, snow_1h, clouds_all, weather_main):
    return {
        "temp_c": temp_c,
        "rain_1h": rain_1h,
        "snow_1h": snow_1h,
        "clouds_all": clouds_all,
        "weather_main": weather_main,
        "weather_description": weather_main.lower(),
        "source": "manual fallback",
    }


def api_window_text() -> str:
    today = datetime.now().date()
    max_forecast = today + pd.Timedelta(days=16).to_pytimedelta()
    return f"API forecast through {max_forecast:%Y-%m-%d}; older dates use archive weather."


def predict_from_row(model, artifacts, row) -> float:
    X = prepare_inference_features(row, artifacts)
    return float(model.predict(X)[0])


def build_24_hour_predictions(
    selected_date: date_type,
    hourly_weather,
    fallback_weather,
    model,
    artifacts,
) -> pd.DataFrame:
    rows = []
    for hour in range(24):
        target_datetime = datetime.combine(selected_date, datetime.min.time()).replace(hour=hour)
        weather = hourly_weather.get(target_datetime, fallback_weather)
        row = build_prediction_row(target_datetime, weather)
        prediction = predict_from_row(model, artifacts, row)
        rows.append({"Hour": f"{hour:02d}:00", "Predicted traffic volume": prediction})
    return pd.DataFrame(rows).set_index("Hour")


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

    :root {
        --card: rgba(15, 23, 42, 0.78);
        --card-border: rgba(148, 163, 184, 0.22);
        --text-soft: #a7b2c5;
    }

    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }

    .stApp {
        background:
            radial-gradient(circle at 15% 15%, rgba(56, 189, 248, 0.16), transparent 28%),
            radial-gradient(circle at 85% 0%, rgba(245, 158, 11, 0.13), transparent 30%),
            linear-gradient(135deg, #07111f 0%, #101827 50%, #111827 100%);
        color: #e5eefb;
    }

    [data-testid="stHeader"] { background: transparent; }

    .block-container {
        padding-top: 0.7rem;
        padding-bottom: 0.8rem;
        max-width: 1280px;
    }

    .hero {
        position: relative;
        padding: 0.85rem 1.1rem;
        border: 1px solid var(--card-border);
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(30,41,59,0.78));
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.28);
        overflow: hidden;
        animation: fadeIn 600ms ease-out;
    }

    .eyebrow {
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.65rem;
        font-weight: 800;
    }

    .hero h1 {
        font-size: clamp(1.45rem, 3vw, 2.45rem);
        line-height: 1;
        margin: 0.2rem 0 0.2rem;
        letter-spacing: -0.05em;
    }

    .hero p {
        color: var(--text-soft);
        font-size: 0.84rem;
        margin: 0;
    }

    .card {
        border: 1px solid var(--card-border);
        border-radius: 18px;
        padding: 0.75rem;
        background: var(--card);
        box-shadow: 0 10px 30px rgba(0,0,0,0.22);
        backdrop-filter: blur(18px);
        animation: fadeInUp 520ms ease-out;
    }

    .card-title {
        color: #e2e8f0;
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        margin-bottom: 0.45rem;
    }

    .result-card { min-height: 235px; }

    .weather-card {
        min-height: 235px;
        position: relative;
        overflow: hidden;
    }

    .weather-card > * {
        position: relative;
        z-index: 2;
    }

    .weather-orb {
        width: 56px;
        height: 56px;
        border-radius: 999px;
        display: grid;
        place-items: center;
        font-size: 0.76rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        background: rgba(255,255,255,0.13);
        box-shadow: 0 0 34px rgba(255,255,255,0.14);
        animation: floaty 4s ease-in-out infinite;
    }

    .weather-card.rainy::before,
    .weather-card.snowy::before,
    .weather-card.foggy::before,
    .weather-card.cloudy::before {
        content: "";
        position: absolute;
        inset: 0;
        opacity: 0.25;
        pointer-events: none;
    }

    .weather-card.rainy::before {
        background-image: repeating-linear-gradient(105deg, rgba(125,211,252,0.0) 0 16px, rgba(125,211,252,0.55) 17px 18px);
        animation: rain 900ms linear infinite;
    }

    .weather-card.snowy::before {
        background-image: radial-gradient(circle, rgba(255,255,255,0.7) 1px, transparent 2px);
        background-size: 24px 24px;
        animation: snow 7s linear infinite;
    }

    .weather-card.foggy::before {
        background: linear-gradient(90deg, transparent, rgba(226,232,240,0.28), transparent);
        animation: fog 6s ease-in-out infinite;
    }

    .weather-card.cloudy::before {
        background: radial-gradient(circle at 25% 40%, rgba(203,213,225,0.34), transparent 18%),
                    radial-gradient(circle at 58% 30%, rgba(148,163,184,0.24), transparent 20%);
        animation: drift 9s ease-in-out infinite alternate;
    }

    .sun-disc {
        position: absolute;
        right: 18px;
        top: 18px;
        width: 64px;
        height: 64px;
        border-radius: 999px;
        background: radial-gradient(circle, #fde68a 0%, #f59e0b 65%, transparent 68%);
        box-shadow: 0 0 42px rgba(245, 158, 11, 0.62);
        animation: sunPulse 2.8s ease-in-out infinite;
        z-index: 1;
    }

    .sun-ray {
        position: absolute;
        right: -30px;
        top: -30px;
        width: 150px;
        height: 150px;
        border-radius: 999px;
        border: 1px solid rgba(253, 230, 138, 0.35);
        animation: spinRay 9s linear infinite;
        z-index: 1;
    }

    .ray-two {
        width: 190px;
        height: 190px;
        animation-duration: 14s;
        animation-direction: reverse;
    }

    .cloud {
        position: absolute;
        width: 120px;
        height: 38px;
        border-radius: 999px;
        background: rgba(226, 232, 240, 0.25);
        filter: blur(1px);
        z-index: 1;
    }

    .cloud::before,
    .cloud::after {
        content: "";
        position: absolute;
        border-radius: 999px;
        background: rgba(226, 232, 240, 0.3);
    }

    .cloud::before {
        width: 54px;
        height: 54px;
        left: 20px;
        top: -24px;
    }

    .cloud::after {
        width: 68px;
        height: 68px;
        right: 12px;
        top: -34px;
    }

    .cloud-one {
        right: -20px;
        top: 52px;
        animation: cloudDrift 8s ease-in-out infinite alternate;
    }

    .cloud-two {
        left: -30px;
        bottom: 42px;
        transform: scale(0.75);
        animation: cloudDrift 11s ease-in-out infinite alternate-reverse;
    }

    .rain-field span,
    .snow-field span {
        position: absolute;
        display: block;
        z-index: 1;
    }

    .rain-field span {
        top: -30px;
        width: 2px;
        height: 48px;
        border-radius: 999px;
        background: rgba(125, 211, 252, 0.75);
        animation: rainDrop 1s linear infinite;
    }

    .rain-field span:nth-child(1) { left: 12%; animation-delay: 0s; }
    .rain-field span:nth-child(2) { left: 28%; animation-delay: 0.18s; }
    .rain-field span:nth-child(3) { left: 45%; animation-delay: 0.34s; }
    .rain-field span:nth-child(4) { left: 61%; animation-delay: 0.08s; }
    .rain-field span:nth-child(5) { left: 78%; animation-delay: 0.22s; }
    .rain-field span:nth-child(6) { left: 90%; animation-delay: 0.42s; }

    .snow-field span {
        top: -20px;
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: rgba(255,255,255,0.86);
        box-shadow: 0 0 12px rgba(255,255,255,0.5);
        animation: snowFall 4.2s linear infinite;
    }

    .snow-field span:nth-child(1) { left: 10%; animation-delay: 0s; }
    .snow-field span:nth-child(2) { left: 26%; animation-delay: 0.7s; }
    .snow-field span:nth-child(3) { left: 42%; animation-delay: 1.4s; }
    .snow-field span:nth-child(4) { left: 62%; animation-delay: 0.3s; }
    .snow-field span:nth-child(5) { left: 78%; animation-delay: 1.1s; }
    .snow-field span:nth-child(6) { left: 92%; animation-delay: 1.9s; }

    .fog-line {
        position: absolute;
        left: -20%;
        width: 140%;
        height: 18px;
        border-radius: 999px;
        background: linear-gradient(90deg, transparent, rgba(226,232,240,0.35), transparent);
        z-index: 1;
        animation: fogSlide 5.5s ease-in-out infinite alternate;
    }

    .fog-one { top: 58px; }
    .fog-two { top: 112px; animation-delay: 0.8s; }
    .fog-three { bottom: 42px; animation-delay: 1.5s; }

    .result-number {
        font-size: clamp(2.3rem, 5vw, 4.2rem);
        font-weight: 800;
        line-height: 0.95;
        letter-spacing: -0.06em;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.42rem 0.65rem;
        font-weight: 800;
        letter-spacing: 0.02em;
    }

    .summary-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.45rem;
        margin-top: 0.7rem;
    }

    .summary-item {
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        padding: 0.5rem;
        background: rgba(15, 23, 42, 0.36);
    }

    .summary-label {
        color: var(--text-soft);
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        font-weight: 800;
    }

    .summary-value {
        color: #f8fafc;
        font-size: 0.9rem;
        font-weight: 800;
        margin-top: 0.1rem;
    }

    .source-pill {
        border-radius: 14px;
        padding: 0.5rem 0.75rem;
        background: rgba(14, 165, 233, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.22);
        color: #bae6fd;
        font-weight: 700;
        font-size: 0.8rem;
    }

    .fallback-pill {
        background: rgba(245, 158, 11, 0.14);
        border-color: rgba(245, 158, 11, 0.3);
        color: #fde68a;
    }

    .compact-note {
        color: var(--text-soft);
        font-size: 0.78rem;
        margin: 0.2rem 0 0;
    }

    .element-container { margin-bottom: 0.32rem; }

    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes drift { from { transform: translateX(-2%) translateY(-1%); } to { transform: translateX(3%) translateY(2%); } }
    @keyframes floaty { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-7px); } }
    @keyframes rain { from { background-position: 0 0; } to { background-position: 0 28px; } }
    @keyframes snow { from { background-position: 0 0; } to { background-position: 0 80px; } }
    @keyframes fog { 0%, 100% { transform: translateX(-35%); } 50% { transform: translateX(35%); } }
    @keyframes sunPulse { 0%, 100% { transform: scale(1); opacity: 0.75; } 50% { transform: scale(1.12); opacity: 1; } }
    @keyframes spinRay { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    @keyframes cloudDrift { from { transform: translateX(-8px); } to { transform: translateX(16px); } }
    @keyframes rainDrop { from { transform: translateY(-45px) rotate(15deg); opacity: 0; } 20% { opacity: 1; } to { transform: translateY(265px) rotate(15deg); opacity: 0; } }
    @keyframes snowFall { from { transform: translateY(-30px) translateX(0); opacity: 0; } 20% { opacity: 1; } to { transform: translateY(260px) translateX(28px); opacity: 0; } }
    @keyframes fogSlide { from { transform: translateX(-8%); opacity: 0.35; } to { transform: translateX(8%); opacity: 0.8; } }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">Metro Interstate I-94</div>
      <h1>Traffic Volume Prediction Dashboard</h1>
      <p>Date + hour in, live weather + saved ML pipeline out. Built for a fast classroom demo.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([0.78, 1.65], gap="medium")

with left:
    st.markdown('<div class="card"><div class="card-title">Prediction Input</div>', unsafe_allow_html=True)
    with st.form("prediction_form"):
        date = st.date_input("Date")
        hour = st.slider("Hour", 0, 23, 8)
        st.caption("Weather is fetched automatically. Fallback is used only if needed.")
        with st.expander("Fallback weather values"):
            temp_c = st.number_input("Temperature (C)", value=15.0)
            rain_1h = st.number_input("Rain in last hour", min_value=0.0, value=0.0)
            snow_1h = st.number_input("Snow in last hour", min_value=0.0, value=0.0)
            clouds_all = st.slider("Cloud cover (%)", 0, 100, 40)
            weather_main = st.selectbox("Weather", ["Clear", "Clouds", "Rain", "Snow", "Mist", "Other"])
        st.caption(api_window_text())
        submitted = st.form_submit_button("Generate Prediction", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

result_slot = right.container()

if submitted:
    if not model_path.exists() or not artifacts_path.exists():
        st.warning("Train the model first with `python run_train.py`.")
    else:
        target_datetime = datetime.combine(date, datetime.min.time()).replace(hour=hour)
        fallback_weather = fallback_weather_values(temp_c, rain_1h, snow_1h, clouds_all, weather_main)

        with st.spinner("Fetching weather and generating prediction..."):
            model, artifacts = load_prediction_assets()
            used_fallback = False
            fallback_error = None
            try:
                hourly_weather = fetch_hourly_weather_for_date(target_datetime)
                weather = hourly_weather.get(target_datetime, fallback_weather)
            except Exception as exc:
                hourly_weather = {}
                weather = fallback_weather
                used_fallback = True
                fallback_error = str(exc)

            row = build_prediction_row(target_datetime, weather)
            prediction = predict_from_row(model, artifacts, row)
            thresholds = load_traffic_thresholds()
            severity = classify_traffic(prediction, thresholds)
            severity_info = SEVERITY_STYLE[severity]
            progress = severity_progress(prediction, thresholds)
            theme = weather_theme(weather.get("weather_main", "Other"))
            weather_fx = weather_effect_markup(theme["class"])
            chart_data = build_24_hour_predictions(date, hourly_weather, fallback_weather, model, artifacts)

        with result_slot:
            weather_col, prediction_col = st.columns([1, 1], gap="medium")

            with weather_col:
                st.markdown(
                    f"""
                    <div class="card weather-card {theme['class']}" style="background: {theme['gradient']};">
                      {weather_fx}
                      <div class="card-title">Weather Summary</div>
                      <div style="display:flex; align-items:center; justify-content:space-between; gap:0.8rem;">
                        <div>
                          <div style="font-size:1.45rem; font-weight:800;">{weather.get('weather_main', 'Other')}</div>
                          <div class="compact-note">{theme['label']}</div>
                        </div>
                        <div class="weather-orb">{theme['icon']}</div>
                      </div>
                      <div class="summary-grid">
                        <div class="summary-item"><div class="summary-label">Temp</div><div class="summary-value">{weather.get('temp_c', 0):.1f} C</div></div>
                        <div class="summary-item"><div class="summary-label">Clouds</div><div class="summary-value">{weather.get('clouds_all', 0)}%</div></div>
                        <div class="summary-item"><div class="summary-label">Rain</div><div class="summary-value">{weather.get('rain_1h', 0):.2f}</div></div>
                        <div class="summary-item"><div class="summary-label">Snow</div><div class="summary-value">{weather.get('snow_1h', 0):.2f}</div></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with prediction_col:
                st.markdown(
                    f"""
                    <div class="card result-card" style="border-color:{severity_info['color']}; box-shadow:0 0 42px {severity_info['soft']};">
                      <div class="card-title">Prediction</div>
                      <div class="result-number">{prediction:,.0f}</div>
                      <div style="margin-top:0.65rem;">
                        <span class="badge" style="background:{severity_info['soft']}; color:{severity_info['color']}; border:1px solid {severity_info['color']};">
                          {severity} traffic
                        </span>
                      </div>
                      <h3 style="margin:0.65rem 0 0.1rem; color:#f8fafc; font-size:1.05rem;">{severity_info['label']}</h3>
                      <p class="compact-note">{severity_info['note']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(progress, text=f"Severity: {severity}")

            source_class = "fallback-pill" if used_fallback else ""
            source_message = "Weather source: Open-Meteo API for Minneapolis I-94."
            if used_fallback:
                source_message = (
                    "Demo fallback weather values are being used because live weather "
                    f"is unavailable for this date. {fallback_error}"
                )
            st.markdown(
                f"""
                <div class="source-pill {source_class}">
                  {source_message} Holiday flag: <strong>{row['holiday']}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="card"><div class="card-title">24-Hour Preview</div>', unsafe_allow_html=True)
            st.line_chart(chart_data, height=165)
            st.markdown("</div>", unsafe_allow_html=True)
else:
    with result_slot:
        st.markdown(
            """
            <div class="card">
              <div class="card-title">Ready for Prediction</div>
              <h2 style="margin-top:0; margin-bottom:0.3rem;">Choose a date and hour.</h2>
              <p class="compact-note">
                Weather, prediction, severity, API status, and 24-hour preview will fit on this screen.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
