import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests

# ---------------- CONFIG ----------------

# If True, dashboard will try to pull live data from the OBD emulator API.
# If the API is not reachable or returns no data, it will fall back to
# simulated fleet data so the UI still works.
USE_EMULATOR = True
EMULATOR_URL = "http://127.0.0.1:8000/telemetry"  # change if API is hosted elsewhere


# ------------- DATA SOURCES -------------


def load_emulator_data() -> pd.DataFrame:
    """
    Pull live telemetry from the OBD telemetry emulator.

    Expected emulator JSON schema (list of dicts):
        {
            "ts": "...ISO timestamp...",
            "vehicle_id": "corolla_2019",
            "rpm": 1500,
            "coolant_temp_c": 90,
            "speed_mph": 45
        }

    This function maps that into the schema that the dashboard uses:
        timestamp, vehicle_id, coolant_temp_f, intake_air_temp_f,
        engine_rpm, boost_psi, vibration_score
    """
    try:
        resp = requests.get(EMULATOR_URL, timeout=3)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        # If anything goes wrong, return empty DF and let caller handle fallback
        return pd.DataFrame()

    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw)

    # Basic safety: make sure required columns exist
    required_cols = {"ts", "vehicle_id", "rpm", "coolant_temp_c", "speed_mph"}
    if not required_cols.issubset(df.columns):
        # Emulator isn't sending what we expect; treat as no data
        return pd.DataFrame()

    # Standardize types & names
    df["timestamp"] = pd.to_datetime(df["ts"])
    df["vehicle_id"] = df["vehicle_id"].fillna("Corolla-2019")

    # Celsius → Fahrenheit
    df["coolant_temp_f"] = df["coolant_temp_c"] * 9.0 / 5.0 + 32.0

    # Derive intake temp roughly below coolant temp so it looks realistic
    df["intake_air_temp_f"] = df["coolant_temp_f"] - 50.0

    # Use real RPM directly
    df["engine_rpm"] = df["rpm"]

    # Make boost and vibration vaguely correlated with RPM / heat so it’s not random noise
    rpm_centered = df["engine_rpm"] - df["engine_rpm"].rolling(30, min_periods=1).mean()
    heat_excess = (df["coolant_temp_f"] - 210).clip(lower=0)

    df["boost_psi"] = 10 + (rpm_centered / 800).clip(-5, 5)
    df["vibration_score"] = (heat_excess / 40 + (rpm_centered / 2000)).clip(lower=0)
    df["vibration_score"] = df["vibration_score"].round(3)

    return df[
        [
            "timestamp",
            "vehicle_id",
            "coolant_temp_f",
            "intake_air_temp_f",
            "engine_rpm",
            "boost_psi",
            "vibration_score",
        ]
    ]


def simulate_fleet_data(num_points: int = 200, num_vehicles: int = 4) -> pd.DataFrame:
    """
    Simulate time-series telemetry for a small fleet.

    Metrics (roughly car / light-duty fleet style):
      - coolant_temp_f (°F)
      - intake_air_temp_f (°F)
      - engine_rpm
      - boost_psi
      - vibration_score (0–1+)
    """
    records = []
    base_time = datetime.now() - timedelta(minutes=num_points)
    timestamps = [base_time + timedelta(minutes=i) for i in range(num_points)]

    for vehicle_id in range(1, num_vehicles + 1):
        # baseline per vehicle
        base_coolant = np.random.uniform(180, 215)
        base_intake = np.random.uniform(90, 100)
        base_rpm = np.random.uniform(1600, 2300)
        base_boost = np.random.uniform(8, 14)
        base_vibration = np.random.uniform(0.25, 0.45)

        # slow drifts to mimic developing issues
        coolant_drift = np.random.uniform(-2, 8)
        intake_drift = coolant_drift * 0.4
        vibration_drift = np.random.uniform(0, 0.3)

        for i, ts in enumerate(timestamps):
            progress = i / num_points

            coolant = base_coolant + progress * coolant_drift + np.random.normal(0, 1.5)
            intake = base_intake + progress * intake_drift + np.random.normal(0, 0.7)
            rpm = base_rpm + np.random.normal(0, 180)
            boost = base_boost + np.random.normal(0, 1.0)
            vibration = (
                base_vibration
                + progress * vibration_drift
                + abs(np.random.normal(0, 0.03))
            )

            records.append(
                {
                    "timestamp": ts,
                    "vehicle_id": f"Unit-{vehicle_id}",
                    "coolant_temp_f": round(coolant, 1),
                    "intake_air_temp_f": round(intake, 1),
                    "engine_rpm": int(rpm),
                    "boost_psi": round(boost, 1),
                    "vibration_score": round(vibration, 3),
                }
            )

    return pd.DataFrame(records)


# --------- HEALTH / RISK LOGIC ---------


def add_health_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple rule-based anomalies and a risk score.

    This is intentionally transparent and interview-friendly:
    easy to walk through on a whiteboard, not "mystery ML."
    """
    df = df.copy()

    # basic thresholds
    df["coolant_high"] = df["coolant_temp_f"] > 250
    df["intake_high"] = df["intake_air_temp_f"] > 200
    df["vibration_high"] = df["vibration_score"] > 5.0

    df = df.sort_values(["vehicle_id", "timestamp"])

    # rolling averages for trend detection
    df["coolant_rolling"] = (
        df.groupby("vehicle_id")["coolant_temp_f"]
        .rolling(window=15, min_periods=5)
        .mean()
        .reset_index(0, drop=True)
    )
    df["intake_rolling"] = (
        df.groupby("vehicle_id")["intake_air_temp_f"]
        .rolling(window=15, min_periods=5)
        .mean()
        .reset_index(0, drop=True)
    )

    df["coolant_trend_high"] = df["coolant_rolling"] > 220
    df["intake_trend_high"] = df["intake_rolling"] > 150

    # rule-based risk score
    df["risk_score"] = (
        df["coolant_high"].astype(int) * 3
        + df["coolant_trend_high"].astype(int) * 2
        + df["intake_high"].astype(int) * 2
        + df["intake_trend_high"].astype(int) * 1
        + df["vibration_high"].astype(int) * 3
    )

    def classify_status(score: int) -> str:
        if score >= 6:
            return "High"
        elif score >= 3:
            return "Medium"
        else:
            return "Low"

    df["risk_band"] = df["risk_score"].apply(classify_status)
    return df


# --------------- STREAMLIT UI ---------------

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🛠️",
    layout="wide",
)

st.title("Predictive Maintenance Lab")
st.caption(
    "Live / simulated fleet telemetry → simple risk scoring → maintenance priorities. "
    "MVP to show how I think about predictive maintenance and asset health."
)

with st.expander("What this dashboard is (and isn’t)", expanded=False):
    st.write(
        """
        **This is an MVP demo**, not a production system:

        - Data can come from a **live OBD-style emulator** or from **synthetic fleet signals**.
        - Risk is scored using **transparent rules**, not a black-box model.
        - The goal is to show how to:
            - ingest and structure time-series telemetry
            - derive health indicators and risk bands
            - surface a **maintenance priority list** for operators.

        In a real deployment, this would plug into actual OBD-II / telematics feeds or equipment sensors.
        """
    )

# Source selector (mostly for debugging/demo)
source_label = "Live emulator" if USE_EMULATOR else "Simulated fleet"
st.sidebar.markdown(f"**Data source:** {source_label}")

# pull data
if USE_EMULATOR:
    df_raw = load_emulator_data()
    if df_raw.empty:
        st.sidebar.warning("No emulator data available, falling back to simulated fleet.")
        df_raw = simulate_fleet_data()
else:
    df_raw = simulate_fleet_data()

if df_raw.empty:
    st.error("No telemetry available from any source.")
    st.stop()

df = add_health_signals(df_raw)

# ---------- SIDEBAR FILTERS ----------

st.sidebar.header("Filters")

vehicle_ids = sorted(df["vehicle_id"].unique())
selected_vehicle = st.sidebar.selectbox("Select unit", vehicle_ids)

risk_filter = st.sidebar.multiselect(
    "Filter by risk band",
    options=["High", "Medium", "Low"],
    default=["High", "Medium", "Low"],
)

# subset based on filters
filtered_df = df[df["vehicle_id"] == selected_vehicle].copy()
fleet_latest = (
    df.sort_values("timestamp")
    .groupby("vehicle_id")
    .tail(1)
    .sort_values("risk_score", ascending=False)
)

if risk_filter:
    fleet_latest = fleet_latest[fleet_latest["risk_band"].isin(risk_filter)]

# ---------- TOP-LEVEL SUMMARY ----------

st.subheader("Fleet health snapshot")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Units monitored", value=len(vehicle_ids))

with col2:
    st.metric(
        "High-risk units",
        value=int((fleet_latest["risk_band"] == "High").sum()),
    )

with col3:
    st.metric(
        "Medium-risk units",
        value=int((fleet_latest["risk_band"] == "Medium").sum()),
    )

st.markdown("### Maintenance priority list")

priority_cols = [
    "vehicle_id",
    "risk_band",
    "risk_score",
    "coolant_temp_f",
    "intake_air_temp_f",
    "vibration_score",
    "timestamp",
]

# sort by “business priority”
fleet_latest = fleet_latest.sort_values(
    ["risk_score", "coolant_temp_f", "intake_air_temp_f", "vehicle_id"],
    ascending=[False, False, False, True],
)

st.dataframe(
    fleet_latest[priority_cols].reset_index(drop=True),
    use_container_width=True,
)

# ---------- DETAIL FOR SELECTED UNIT ----------

st.markdown(f"### Unit detail: `{selected_vehicle}`")

unit_df = filtered_df.sort_values("timestamp")
latest = unit_df.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Risk band", latest["risk_band"])
with c2:
    st.metric("Coolant temp (°F)", f"{latest['coolant_temp_f']:.1f}")
with c3:
    st.metric("Intake air (°F)", f"{latest['intake_air_temp_f']:.1f}")
with c4:
    st.metric("Vibration score", f"{latest['vibration_score']:.2f}")

st.markdown("#### Trend lines")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.line_chart(
        unit_df.set_index("timestamp")[["coolant_temp_f", "intake_air_temp_f"]],
        height=260,
    )

with chart_col2:
    st.line_chart(
        unit_df.set_index("timestamp")[["vibration_score"]],
        height=260,
    )

st.markdown("#### Recent raw telemetry")

st.dataframe(
    unit_df[
        [
            "timestamp",
            "coolant_temp_f",
            "intake_air_temp_f",
            "engine_rpm",
            "boost_psi",
            "vibration_score",
            "risk_score",
            "risk_band",
        ]
    ].tail(50),
    use_container_width=True,
)
