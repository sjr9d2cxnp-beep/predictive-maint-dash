import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---------- DATA SIMULATION ----------

def simulate_fleet_data(num_points: int = 200, num_vehicles: int = 4) -> pd.DataFrame:
    """
    Simulate time-series telemetry for a small fleet.

    Metrics (roughly car / light-duty fleet style):
      - coolant_temp_f (°F)
      - intake_air_temp_f (°F)
      - engine_rpm
      - boost_psi
      - vibration_score (0–1)
    """
    records = []
    # generate timestamps spaced 1 minute apart, oldest -> newest
    base_time = datetime.now() - timedelta(minutes=num_points)
    timestamps = [base_time + timedelta(minutes=i) for i in range(num_points)]

    for vehicle_id in range(1, num_vehicles + 1):
        # base values per vehicle (slight variation)
        base_coolant = np.random.uniform(120, 200)         # normal highway range
        base_intake = np.random.uniform(90, 100)
        base_rpm = np.random.uniform(1700, 2300)
        base_boost = np.random.uniform(8, 14)
        base_vibration = np.random.uniform(0.25, 0.45)

        # possible slow drift to simulate developing fault
        coolant_drift = np.random.uniform(-2, 6)
        intake_drift = coolant_drift * 0.5
        vibration_drift = np.random.uniform(0, 0.25)

        for i, ts in enumerate(timestamps):
            progress = i / num_points

            coolant = base_coolant + progress * coolant_drift + np.random.normal(0, 1.2)
            intake = base_intake + progress * intake_drift + np.random.normal(0, 0.6)
            rpm = base_rpm + np.random.normal(0, 180)
            boost = base_boost + np.random.normal(0, 1.0)
            vibration = base_vibration + progress * vibration_drift + abs(np.random.normal(0, 0.03))

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

    df = pd.DataFrame(records)
    return df


def add_health_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple rule-based anomalies and a risk score.
    This is intentionally simple: MVP to show thinking, not hardcore ML.
    """
    df = df.copy()

    # basic rule thresholds (tweak these as needed)
    df["coolant_high"] = df["coolant_temp_f"] > 250
    df["intake_high"] = df["intake_air_temp_f"] > 200
    df["vibration_high"] = df["vibration_score"] > 5.00

    # rolling “hot trend” flags (moving average creeping up)
    df = df.sort_values(["vehicle_id", "timestamp"])
    df["coolant_rolling"] = (
        df.groupby("vehicle_id")["coolant_temp_f"].rolling(window=15, min_periods=5).mean().reset_index(0, drop=True)
    )
    df["intake_rolling"] = (
        df.groupby("vehicle_id")["intake_air_temp_f"].rolling(window=15, min_periods=5).mean().reset_index(0, drop=True)
    )

    df["coolant_trend_high"] = df["coolant_rolling"] > 220
    df["intake_trend_high"] = df["intake_rolling"] > 150

    # create a simple risk score
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


# ---------- STREAMLIT UI ----------

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🛠️",
    layout="wide",
)

st.title("Predictive Maintenance Lab")
st.caption(
    "Simulated fleet telemetry → simple risk scoring → maintenance priorities. "
    "MVP to demonstrate how I think about predictive maintenance and asset health."
)

with st.expander("What this dashboard is (and isn’t)", expanded=False):
    st.write(
        """
        **This is an MVP demo**, not a production system:

        - Data is **synthetic** but modeled on realistic ranges for light-duty vehicles.
        - Risk is scored using **transparent rules**, not a black-box ML model.
        - The goal is to show how to:
            - ingest and structure time-series signals
            - derive simple health indicators and risk bands
            - surface a **maintenance priority list** for operators.

        In a real deployment, this would plug into actual OBD-II / telematics feeds or equipment sensors.
        """
    )

# simulate data & add health metrics
df_raw = simulate_fleet_data()
df = add_health_signals(df_raw)

# sidebar filters
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
    st.metric(
        "Units monitored",
        value=len(vehicle_ids),
    )

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
# Sort by business-meaningful priority
fleet_latest = fleet_latest.sort_values(
    ["risk_score", "coolant_temp_f", "intake_air_temp_f"],
    ascending=[False, False, False]
)
fleet_latest = fleet_latest.sort_values(
    ["risk_score", "coolant_temp_f", "intake_air_temp_f", "vehicle_id"],
    ascending=[False, False, False, True]
)

st.dataframe(
    fleet_latest[priority_cols].reset_index(drop=True),
    use_container_width=True,
)


# ---------- DETAILED VIEW FOR SELECTED UNIT ----------

st.markdown(f"### Unit detail: `{selected_vehicle}`")

unit_df = filtered_df.sort_values("timestamp")

# latest reading for selected unit
latest = unit_df.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Risk band", latest["risk_band"], help="Rule-based risk classification for this unit.")
with c2:
    st.metric("Coolant temp (°F)", f"{latest['coolant_temp_f']:.1f}")
with c3:
    st.metric("Intake air (°F)", f"{latest['intake_air_temp_f']:.1f}")
with c4:
    st.metric("Vibration score", f"{latest['vibration_score']:.2f}")

# charts
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

st.markdown("#### Raw telemetry (recent)")

st.dataframe(
    unit_df[[
        "timestamp",
        "coolant_temp_f",
        "intake_air_temp_f",
        "engine_rpm",
        "boost_psi",
        "vibration_score",
        "risk_score",
        "risk_band",
    ]].tail(50),
    use_container_width=True,
)
