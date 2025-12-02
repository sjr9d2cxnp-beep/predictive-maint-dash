# Predictive Maintenance Dashboard – Connected Ops Lab Module

This project is one module in my **Connected Operations Lab**, focused on turning raw equipment signals into clear, risk-based maintenance decisions.

## What this simulates

- Assets: small fleet of vehicles / equipment
- Input: simulated telemetry (temperature, RPM, vibration, etc.)
- Output: a dashboard that:
  - scores each asset for maintenance risk
  - highlights outliers and emerging issues
  - surfaces a prioritized list of assets to inspect

The emphasis isn’t “fancy ML,” it’s **clarity and workflows**:

- show how telemetry becomes a structured dataset
- apply simple, explainable rules / scoring for maintenance risk
- present the results so non-technical stakeholders can act on it

## Architecture (MVP)

1. **Data simulation**
   - Python function generates time-series telemetry for multiple vehicles
   - Metrics like coolant_temp_f, engine_rpm, vibration_score, etc.

2. **Scoring logic**
   - Thresholds / heuristics for “normal” vs “risky”
   - Per-asset health score and risk category (Low / Medium / High)

3. **UI layer**
   - Time-series chart for key metrics
   - Prioritized list of highest-risk assets
   - Drill-down into a single asset’s history

## Example workflow

1. Generate telemetry for a small fleet  
2. Calculate maintenance risk scores  
3. View:
   - top-N assets by risk
   - how risk evolves over time
   - which signals contributed most to the score

This mirrors how a Sales Engineer would walk a customer through a predictive maintenance story: **from raw data → to signals → to decisions.**

## How this fits into the Connected Ops Lab

This dashboard is designed to plug into my broader lab:

- Telemetry Emulator: IoT → API → dashboard
- Predictive Maintenance Dashboard: this module
- Future: anomaly alerts + notifications + integrated “maintenance queue”

As I wire these together, the goal is an end-to-end demo that shows how connected operations tools create **operational clarity** for frontline teams and leadership.

To run:

Open terminal

Run the following commands:

git clone https://github.com/sjr9d2cxnp-beep/predictive-maint-dash.git

python -m venv .venv

.venv\Scripts\activate

If it is blocked, run this ONCE as an admin in Powershell and then try again:

Set-ExecutionPolicy -ExecutionPolicyRemoteSigned -Scope CurrentUser

pip install -r requirements.txt

streamlit run app.py



