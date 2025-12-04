import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

API_URL = "http://127.0.0.1:8000/telemetry"


# ---------- DATA HELPERS ----------

def simulate_fleet_data(num_points: int = 600, num_vehicles: int = 4) -> pd.DataFrame:
    """
    Fallback: generate synthetic fleet data with the same schema as the live feed,
    tuned to feel like light-duty vehicles cruising at highway speeds.
    """
    records = []
    base_time = datetime.now() - timedelta(seconds=num_points)
    timestamps = [base_time + timedelta(seconds=i) for i in range(num_points)]

    for vehicle_id in range(1, num_vehicles + 1):
        # realistic cruise baselines
        base_coolant = np.random.uniform(188, 202)     # around 195°F
        base_intake = np.random.uniform(60, 70)        # ambient-ish
        base_rpm = np.random.uniform(1850, 2050)
        base_speed = np.random.uniform(65, 75)
        base_vibration = np.random.uniform(0.25, 0.45)

        coolant_drift = np.random.uniform(-0.5, 2.0)
        intake_drift = coolant_drift * 0.4
        vibration_drift = np.random.uniform(0.0, 0.10)

        hours = 0.0

        for i, ts in enumerate(timestamps):
            progress = i / num_points

            rpm = base_rpm + np.random.normal(0, 60)
            rpm = np.clip(rpm, 1800, 2100)

            speed = base_speed + np.random.normal(0, 2.0)
            speed = np.clip(speed, 60, 80)

            coolant = base_coolant + progress * coolant_drift + np.random.normal(0, 1.0)
            intake = (
                base_intake
                + progress * intake_drift
                + (rpm - 1950) / 80.0
                + np.random.normal(0, 0.8)
            )

            vibration = (
                base_vibration
                + progress * vibration_drift
                + abs(np.random.normal(0, 0.02))
            )

            # occasional synthetic overheat to match emulator behavior
            if np.random.rand() < 0.02:
                coolant = np.random.uniform(225, 235)
                intake += np.random.uniform(8, 15)
                vibration += 0.7

            hours += 1.0 / 3600.0

            records.append(
                {
                    "timestamp": ts,
                    "vehicle_id": f"Unit-{vehicle_id}",
                    "coolant_temp_f": round(coolant, 1),
                    "intake_air_temp_f": round(intake, 1),
                    "engine_rpm": int(rpm),
                    "speed_mph": round(speed, 1),
                    "vibration_score": round(vibration, 3),
                    "engine_hours": round(hours, 2),
                }
            )

    return pd.DataFrame(records)


def load_live_telemetry(api_url: str = API_URL) -> pd.DataFrame:
    """Fetch recent telemetry points from the FastAPI service."""
    try:
        resp = requests.get(api_url, timeout=2)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if "ts" in df.columns:
            df["timestamp"] = pd.to_datetime(df["ts"])
            df = df.drop(columns=["ts"])

        return df
    except Exception:
        # If API is down or empty, caller will fall back to simulation.
        return pd.DataFrame()


def add_health_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rule-based anomalies and a risk score tuned for realistic Corolla-like telemetry.
    """
    df = df.copy()

    # --- HARD THRESHOLDS (per-sample) ---
    # normal coolant ~195°F; 220+ is bad, 230 is definitely a problem
    df["coolant_high"] = df["coolant_temp_f"] > 220

    # intake should live around 60–80°F; 120+ is very abnormal
    df["intake_high"] = df["intake_air_temp_f"] > 120

    # vibration baseline ~0.3; 1.0+ indicates roughness
    df["vibration_high"] = df["vibration_score"] > 1.0

    # --- TREND FLAGS (rolling mean) ---

    df = df.sort_values(["vehicle_id", "timestamp"])

    # about 10 minutes at 1 Hz
    df["coolant_rolling"] = (
        df.groupby("vehicle_id")["coolant_temp_f"]
        .rolling(window=600, min_periods=60)
        .mean()
        .reset_index(0, drop=True)
    )
    df["intake_rolling"] = (
        df.groupby("vehicle_id")["intake_air_temp_f"]
        .rolling(window=600, min_periods=60)
        .mean()
        .reset_index(0, drop=True)
    )

    # persistent coolant > 210°F is suspicious even if individual spikes are lower
    df["coolant_trend_high"] = df["coolant_rolling"] > 210
    df["intake_trend_high"] = df["intake_rolling"] > 90

    # --- RISK SCORE ---

    df["risk_score"] = (
        df["coolant_high"].astype(int) * 3
        + df["coolant_trend_high"].astype(int) * 2
        + df["intake_high"].astype(int) * 2
        + df["intake_trend_high"].astype(int) * 1
        + df["vibration_high"].astype(int) * 3
    )

    def classify(score: int) -> str:
        if score >= 6:
            return "High"
        elif score >= 3:
            return "Medium"
        else:
            return "Low"

    df["risk_band"] = df["risk_score"].apply(classify)
    return df


def load_data() -> tuple[pd.DataFrame, str]:
    """Try live data, fall back to simulation. Return (df, source)."""
    live_df = load_live_telemetry()
    if live_df.empty:
        df_raw = simulate_fleet_data()
        source = "simulated"
    else:
        df_raw = live_df
        source = "live"

    df = add_health_signals(df_raw)
    return df, source


# ---------- STREAMLIT UI ----------

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🛠️",
    layout="wide",
)

# Simple auto-refresh every 15 seconds
st.markdown(
    """
    <meta http-equiv="refresh" content="15">
    """,
    unsafe_allow_html=True,
)

st.title("Predictive Maintenance Lab")

st.caption(
    "Live / simulated fleet telemetry → rule-based risk scoring → maintenance priorities. "
    "MVP to show how I think about predictive maintenance and asset health."
)

with st.expander("What this dashboard is (and isn’t)", expanded=False):
    st.write(
        """
        **This is an MVP demo**, not a production system:

        - Data can come from a **live FastAPI telemetry feed** (OBD-II emulator) or a simulated fleet.
        - Risk is scored using **transparent rules**, not a black-box model.
        - The focus is on:
            - ingesting and structuring time-series signals
            - deriving health indicators and risk bands
            - surfacing a **maintenance priority list**.

        In a real deployment, the live API would sit behind IoT gateways / telematics devices
        and feed real vehicle or equipment data.
        """
    )

df, source = load_data()
if df.empty:
    st.error("No telemetry data available yet.")
    st.stop()

vehicle_ids = sorted(df["vehicle_id"].unique())
latest_ts = df["timestamp"].max()

st.info(f"Data source: **{source}** • Last sample: `{latest_ts}`")

# sidebar filters
st.sidebar.header("Filters")

selected_vehicle = st.sidebar.selectbox("Select unit", vehicle_ids)

risk_filter = st.sidebar.multiselect(
    "Filter by risk band",
    options=["High", "Medium", "Low"],
    default=["High", "Medium", "Low"],
)

# subset based on filters
unit_df = df[df["vehicle_id"] == selected_vehicle].sort_values("timestamp")

fleet_latest = (
    df.sort_values("timestamp")
    .groupby("vehicle_id")
    .tail(1)
)

if risk_filter:
    fleet_latest = fleet_latest[fleet_latest["risk_band"].isin(risk_filter)]

# ---------- TOP-LEVEL SUMMARY ----------

st.subheader("Fleet health snapshot")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Units monitored", value=len(vehicle_ids))

with col2:
    st.metric("High-risk units", value=int((fleet_latest["risk_band"] == "High").sum()))

with col3:
    st.metric("Medium-risk units", value=int((fleet_latest["risk_band"] == "Medium").sum()))

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

fleet_latest = fleet_latest.sort_values(
    ["risk_score", "coolant_temp_f", "intake_air_temp_f", "vehicle_id"],
    ascending=[False, False, False, True],
)

st.dataframe(
    fleet_latest[priority_cols].reset_index(drop=True),
    use_container_width=True,
)

# ---------- DETAILED VIEW FOR SELECTED UNIT ----------

st.markdown(f"### Unit detail: `{selected_vehicle}`")

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

c5, c6 = st.columns(2)
with c5:
    st.metric("Engine RPM", f"{latest['engine_rpm']}")
with c6:
    st.metric("Engine hours", f"{latest['engine_hours']:.2f}")

st.markdown("#### Trend lines")

# Slight smoothing for nicer-looking cruising lines
plot_df = unit_df.set_index("timestamp").copy()
plot_df["coolant_temp_smooth"] = plot_df["coolant_temp_f"].rolling(window=20, min_periods=1).mean()
plot_df["intake_air_smooth"] = plot_df["intake_air_temp_f"].rolling(window=20, min_periods=1).mean()
plot_df["vibration_smooth"] = plot_df["vibration_score"].rolling(window=20, min_periods=1).mean()

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.line_chart(
        plot_df[["coolant_temp_smooth", "intake_air_smooth"]],
        height=260,
    )

with chart_col2:
    st.line_chart(
        plot_df[["vibration_smooth"]],
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
            "speed_mph",
            "vibration_score",
            "engine_hours",
            "risk_score",
            "risk_band",
        ]
    ].tail(100),
    use_container_width=True,
)
