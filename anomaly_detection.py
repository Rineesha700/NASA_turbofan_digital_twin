import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "twin_state"
    / "fleet_telemetry_history.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "twin_state"
    / "fleet_anomaly_history.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "anomaly_detector.pkl"
)


# ---------------------------------------------------------
# SENSOR COLUMNS
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("=" * 80)
print("ANOMALY DETECTION")
print("=" * 80)

print("\nLoading fleet telemetry...")

df = pd.read_csv(INPUT_PATH)

print(f"Records: {len(df)}")
print(f"Engines: {df['engine_id'].nunique()}")


# ---------------------------------------------------------
# PREPARE SENSOR DATA
# ---------------------------------------------------------

X = df[sensor_columns].copy()

X = X.fillna(X.median())


# ---------------------------------------------------------
# TRAIN ISOLATION FOREST
# ---------------------------------------------------------

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

model.fit(X)


# ---------------------------------------------------------
# PREDICT ANOMALIES
# ---------------------------------------------------------

df["anomaly_prediction"] = model.predict(X)

df["anomaly_score"] = model.decision_function(X)


df["anomaly_status"] = df[
    "anomaly_prediction"
].map(
    {
        1: "NORMAL",
        -1: "ANOMALY"
    }
)


# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------

joblib.dump(
    model,
    MODEL_PATH
)

print("\nAnomaly detection model saved:")
print(MODEL_PATH)


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nAnomaly results saved:")
print(OUTPUT_PATH)


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("ANOMALY DETECTION COMPLETED")
print("=" * 80)

print("\nAnomaly Summary:")

print(
    df["anomaly_status"]
    .value_counts()
    .to_string()
)


# ---------------------------------------------------------
# ENGINE SUMMARY
# ---------------------------------------------------------

print("\nAnomalies by Engine:")

engine_anomalies = (
    df[df["anomaly_status"] == "ANOMALY"]
    .groupby("engine_id")
    .size()
    .sort_values(ascending=False)
)

print(
    engine_anomalies.head(10).to_string()
)