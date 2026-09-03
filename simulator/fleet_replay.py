import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from digital_twin.digital_twin import DigitalTwin


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "test_FD001.txt"
MODEL_PATH = PROJECT_ROOT / "models" / "rul_histgradientboosting.pkl"
FEATURE_PATH = PROJECT_ROOT / "models" / "feature_columns.pkl"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "twin_state"
    / "fleet_telemetry_history.csv"
)


# ---------------------------------------------------------
# NASA C-MAPSS COLUMNS
# ---------------------------------------------------------

columns = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]


# ---------------------------------------------------------
# MODEL SENSORS
# ---------------------------------------------------------

model_sensors = [
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
    "sensor_21",
]


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_telemetry():

    return pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=columns
    )


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

def load_model():

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)

    return model, feature_columns


# ---------------------------------------------------------
# CREATE MODEL FEATURES
# ---------------------------------------------------------

def create_features(history):

    features = history.copy()

    for sensor in model_sensors:

        features[f"{sensor}_mean_5"] = (
            features.groupby("engine_id")[sensor]
            .transform(
                lambda x:
                x.rolling(
                    window=5,
                    min_periods=1
                ).mean()
            )
        )

        features[f"{sensor}_std_5"] = (
            features.groupby("engine_id")[sensor]
            .transform(
                lambda x:
                x.rolling(
                    window=5,
                    min_periods=1
                ).std()
            )
        )

    return features.fillna(0)


# ---------------------------------------------------------
# HEALTH CLASSIFICATION
# ---------------------------------------------------------

def get_health_status(rul):

    if rul > 80:
        return "HEALTHY"

    elif rul > 30:
        return "WARNING"

    else:
        return "CRITICAL"


# ---------------------------------------------------------
# PROCESS ONE ENGINE
# ---------------------------------------------------------

def process_engine(
    telemetry,
    engine_id,
    model,
    feature_columns
):

    engine_data = (
        telemetry[
            telemetry["engine_id"] == engine_id
        ]
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    if engine_data.empty:
        return []

    twin = DigitalTwin(int(engine_id))

    history = []
    records = []

    for _, row in engine_data.iterrows():

        history.append(row.to_dict())

        history_df = pd.DataFrame(history)

        feature_data = create_features(history_df)

        current_features = feature_data.iloc[[-1]]

        X_current = current_features[
            feature_columns
        ]

        predicted_rul = model.predict(
            X_current
        )[0]

        predicted_rul = max(
            0,
            predicted_rul
        )

        health = get_health_status(
            predicted_rul
        )

        sensor_data = {
            sensor: float(row[sensor])
            for sensor in model_sensors
        }

        twin.update(
            cycle=int(row["cycle"]),
            sensor_data=sensor_data,
            predicted_rul=float(predicted_rul),
            health_status=health
        )

        record = {
            "engine_id": int(row["engine_id"]),
            "cycle": int(row["cycle"]),
        }

        record.update(sensor_data)

        record["predicted_rul"] = float(
            predicted_rul
        )

        record["health_status"] = health

        records.append(record)

    return records


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 80)
    print("NASA TURBOFAN FLEET DIGITAL TWIN")
    print("=" * 80)

    print("\nLoading NASA telemetry...")

    telemetry = load_telemetry()

    print(
        f"Records: {len(telemetry)}"
    )

    print(
        f"Engines: {telemetry['engine_id'].nunique()}"
    )

    print("\nLoading trained RUL model...")

    model, feature_columns = load_model()

    print("Model loaded successfully!")

    print(
        f"Model features: {len(feature_columns)}"
    )

    all_records = []

    engine_ids = sorted(
        telemetry["engine_id"].unique()
    )

    print("\nProcessing fleet...\n")

    for engine_id in engine_ids:

        records = process_engine(
            telemetry,
            engine_id,
            model,
            feature_columns
        )

        all_records.extend(records)

        latest = records[-1]

        print(
            f"Engine {engine_id:3d} | "
            f"Cycles: {len(records):3d} | "
            f"RUL: {latest['predicted_rul']:7.2f} | "
            f"Health: {latest['health_status']}"
        )

    # -----------------------------------------------------
    # SAVE FLEET TELEMETRY
    # -----------------------------------------------------

    fleet_df = pd.DataFrame(
        all_records
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fleet_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 80)
    print("FLEET REPLAY COMPLETED")
    print("=" * 80)

    print(
        f"\nTotal telemetry records: {len(fleet_df)}"
    )

    print(
        f"Total engines: {fleet_df['engine_id'].nunique()}"
    )

    print(
        f"\nFleet telemetry saved to:"
    )

    print(OUTPUT_PATH)

    # -----------------------------------------------------
    # FLEET HEALTH SUMMARY
    # -----------------------------------------------------

    latest_states = (
        fleet_df
        .sort_values("cycle")
        .groupby("engine_id")
        .tail(1)
    )

    print("\nFleet Health Summary:")

    print(
        latest_states[
            "health_status"
        ]
        .value_counts()
        .to_string()
    )