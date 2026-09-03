import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# DATA PATH
# ---------------------------------------------------------

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "twin_state"
    / "fleet_maintenance_history.csv"
)


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="NASA Turbofan Digital Twin",
    page_icon="✈️",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("✈️ NASA Turbofan Digital Twin")

st.markdown(
    "Fleet health monitoring, RUL prediction, anomaly detection "
    "and predictive maintenance"
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

if not DATA_PATH.exists():

    st.error(
        "Maintenance telemetry data not found. "
        "Please run the anomaly detection and maintenance scripts first."
    )

    st.stop()


@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


df = load_data()


# ---------------------------------------------------------
# LATEST STATE OF EACH ENGINE
# ---------------------------------------------------------

latest_states = (
    df
    .sort_values(["engine_id", "cycle"])
    .groupby("engine_id")
    .tail(1)
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# FLEET COUNTS
# ---------------------------------------------------------

total_engines = latest_states["engine_id"].nunique()

healthy_count = (
    latest_states["health_status"] == "HEALTHY"
).sum()

warning_count = (
    latest_states["health_status"] == "WARNING"
).sum()

critical_count = (
    latest_states["health_status"] == "CRITICAL"
).sum()

anomaly_count = (
    latest_states["anomaly_status"] == "ANOMALY"
).sum()

maintenance_count = (
    latest_states["maintenance_decision"]
    == "MAINTENANCE REQUIRED"
).sum()

inspection_count = (
    latest_states["maintenance_decision"]
    == "INSPECTION RECOMMENDED"
).sum()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("Digital Twin Controls")

engine_ids = sorted(
    latest_states["engine_id"].unique()
)

default_index = (
    engine_ids.index(49)
    if 49 in engine_ids
    else 0
)

selected_engine = st.sidebar.selectbox(
    "Select Engine",
    engine_ids,
    index=default_index
)


# ---------------------------------------------------------
# FLEET OVERVIEW
# ---------------------------------------------------------

st.header("Fleet Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Engines",
        total_engines
    )

with col2:
    st.metric(
        "Healthy",
        healthy_count
    )

with col3:
    st.metric(
        "Warning",
        warning_count
    )

with col4:
    st.metric(
        "Critical",
        critical_count
    )


# ---------------------------------------------------------
# FLEET VISUALIZATIONS
# ---------------------------------------------------------

fleet_col1, fleet_col2 = st.columns(2)


with fleet_col1:

    st.subheader("Fleet Health Distribution")

    health_counts = (
        latest_states["health_status"]
        .value_counts()
        .reindex(
            ["HEALTHY", "WARNING", "CRITICAL"],
            fill_value=0
        )
        .reset_index()
    )

    health_counts.columns = [
        "Health Status",
        "Engines"
    ]

    fig_health = px.bar(
        health_counts,
        x="Health Status",
        y="Engines",
        text="Engines"
    )

    fig_health.update_layout(
        height=400,
        xaxis_title="Health Status",
        yaxis_title="Number of Engines"
    )

    fig_health.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig_health,
        use_container_width=True
    )


with fleet_col2:

    st.subheader("Fleet RUL Distribution")

    fig_rul_fleet = px.histogram(
        latest_states,
        x="predicted_rul",
        nbins=20,
        labels={
            "predicted_rul": "Predicted RUL (cycles)"
        }
    )

    fig_rul_fleet.update_layout(
        height=400,
        xaxis_title="Predicted RUL (cycles)",
        yaxis_title="Number of Engines"
    )

    st.plotly_chart(
        fig_rul_fleet,
        use_container_width=True
    )


# ---------------------------------------------------------
# MAINTENANCE OVERVIEW
# ---------------------------------------------------------

st.subheader("Maintenance Overview")

maintenance_col1, maintenance_col2, maintenance_col3 = st.columns(3)

with maintenance_col1:

    st.metric(
        "Monitor",
        total_engines - inspection_count - maintenance_count
    )

with maintenance_col2:

    st.metric(
        "Inspection Recommended",
        inspection_count
    )

with maintenance_col3:

    st.metric(
        "Maintenance Required",
        maintenance_count
    )


# ---------------------------------------------------------
# FLEET ENGINE TABLE
# ---------------------------------------------------------

st.subheader("Fleet Engine Status")

fleet_table = latest_states[
    [
        "engine_id",
        "cycle",
        "predicted_rul",
        "health_status",
        "anomaly_status",
        "maintenance_decision"
    ]
].copy()

fleet_table.columns = [
    "Engine",
    "Current Cycle",
    "Predicted RUL",
    "Health",
    "Anomaly",
    "Maintenance Decision"
]

fleet_table["Predicted RUL"] = (
    fleet_table["Predicted RUL"]
    .round(1)
)

fleet_table = fleet_table.sort_values(
    "Predicted RUL"
)

st.dataframe(
    fleet_table,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# SELECTED ENGINE
# ---------------------------------------------------------

engine_df = (
    df[
        df["engine_id"] == selected_engine
    ]
    .sort_values("cycle")
    .reset_index(drop=True)
)


if engine_df.empty:

    st.warning(
        "No telemetry available for this engine."
    )

    st.stop()


latest = engine_df.iloc[-1]


st.divider()

st.header(
    f"Engine {selected_engine} — Digital Twin"
)


# ---------------------------------------------------------
# ENGINE STATUS
# ---------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Current Cycle",
        int(latest["cycle"])
    )

with col2:

    st.metric(
        "Predicted RUL",
        f"{latest['predicted_rul']:.1f} cycles"
    )

with col3:

    st.metric(
        "Health",
        latest["health_status"]
    )

with col4:

    st.metric(
        "Anomaly",
        latest["anomaly_status"]
    )


# ---------------------------------------------------------
# MAINTENANCE RECOMMENDATION
# ---------------------------------------------------------

st.subheader("Maintenance Recommendation")

decision = latest["maintenance_decision"]

reason = latest["maintenance_reason"]


if decision == "MAINTENANCE REQUIRED":

    st.error(
        f"🔴 {decision}"
    )

elif decision == "INSPECTION RECOMMENDED":

    st.warning(
        f"🟡 {decision}"
    )

else:

    st.success(
        f"🟢 {decision}"
    )


st.write(
    f"**Reason:** {reason}"
)


# ---------------------------------------------------------
# RUL TREND
# ---------------------------------------------------------

st.subheader("RUL Degradation Trend")

fig_rul = px.line(
    engine_df,
    x="cycle",
    y="predicted_rul",
    markers=True,
    labels={
        "cycle": "Engine Cycle",
        "predicted_rul": "Predicted RUL (cycles)"
    }
)

fig_rul.add_hline(
    y=80,
    line_dash="dash",
    annotation_text="Warning threshold"
)

fig_rul.add_hline(
    y=30,
    line_dash="dash",
    annotation_text="Critical threshold"
)

fig_rul.update_layout(
    height=450,
    hovermode="x unified"
)

st.plotly_chart(
    fig_rul,
    use_container_width=True
)


# ---------------------------------------------------------
# ANOMALY TREND
# ---------------------------------------------------------

st.subheader("Anomaly Detection")

anomaly_df = engine_df.copy()

anomaly_df["anomaly_value"] = (
    anomaly_df["anomaly_status"]
    .map(
        {
            "NORMAL": 0,
            "ANOMALY": 1
        }
    )
)

fig_anomaly = px.scatter(
    anomaly_df,
    x="cycle",
    y="anomaly_score",
    color="anomaly_status",
    hover_data=[
        "predicted_rul",
        "health_status"
    ],
    labels={
        "cycle": "Engine Cycle",
        "anomaly_score": "Anomaly Score"
    }
)

fig_anomaly.update_layout(
    height=400,
    hovermode="closest"
)

st.plotly_chart(
    fig_anomaly,
    use_container_width=True
)


# ---------------------------------------------------------
# SENSOR TELEMETRY
# ---------------------------------------------------------

st.subheader("Sensor Telemetry")

sensor_columns = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21"
]

available_sensors = [
    sensor
    for sensor in sensor_columns
    if sensor in engine_df.columns
]

selected_sensor = st.selectbox(
    "Select Sensor",
    available_sensors
)

fig_sensor = px.line(
    engine_df,
    x="cycle",
    y=selected_sensor,
    markers=True,
    labels={
        "cycle": "Engine Cycle",
        selected_sensor: selected_sensor
    }
)

fig_sensor.update_layout(
    height=400,
    hovermode="x unified"
)

st.plotly_chart(
    fig_sensor,
    use_container_width=True
)


# ---------------------------------------------------------
# LATEST TELEMETRY
# ---------------------------------------------------------

st.subheader("Latest Engine Telemetry")

telemetry_display = {
    sensor: latest[sensor]
    for sensor in available_sensors
}

telemetry_display["Predicted RUL"] = (
    latest["predicted_rul"]
)

telemetry_display["Health Status"] = (
    latest["health_status"]
)

telemetry_display["Anomaly Status"] = (
    latest["anomaly_status"]
)

telemetry_df = pd.DataFrame(
    telemetry_display.items(),
    columns=["Parameter", "Value"]
)

st.dataframe(
    telemetry_df,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# DIGITAL TWIN INFORMATION
# ---------------------------------------------------------

st.subheader("Digital Twin Information")

info_col1, info_col2 = st.columns(2)

with info_col1:

    st.write(
        "**Engine ID:**",
        selected_engine
    )

    st.write(
        "**Cycles Available:**",
        len(engine_df)
    )

with info_col2:

    st.write(
        "**Latest Cycle:**",
        int(latest["cycle"])
    )

    st.write(
        "**Maintenance Decision:**",
        decision
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "NASA C-MAPSS Turbofan Digital Twin | "
    "RUL Prediction • Anomaly Detection • Predictive Maintenance"
)