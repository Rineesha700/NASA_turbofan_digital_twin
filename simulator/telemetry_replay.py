import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow Python to import modules from the project root
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import time
import joblib
import pandas as pd

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from digital_twin.digital_twin import DigitalTwin


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "test_FD001.txt"
MODEL_PATH = PROJECT_ROOT / "models" / "rul_histgradientboosting.pkl"
FEATURE_PATH = PROJECT_ROOT / "models" / "feature_columns.pkl"

TELEMETRY_HISTORY_PATH = (
    PROJECT_ROOT / "data" / "twin_state" / "telemetry_history.csv"
)


# ============================================================
# AZURE STORAGE CONFIGURATION
# ============================================================

STORAGE_ACCOUNT_NAME = "nasatwindata2026"
CONTAINER_NAME = "telemetry"
BLOB_NAME = "telemetry_history.csv"


# ============================================================
# NASA C-MAPSS COLUMN NAMES
# ============================================================

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
    "sensor_21"
]


# ============================================================
# SENSORS USED BY THE ML MODEL
# ============================================================

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
    "sensor_21"
]


# ============================================================
# LOAD NASA TELEMETRY
# ============================================================

def load_telemetry():

    data = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=columns
    )

    return data


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)

    return model, feature_columns


# ============================================================
# CREATE MODEL FEATURES
# ============================================================

def create_features(history):

    features = history.copy()

    for sensor in model_sensors:

        # 5-cycle rolling mean
        features[f"{sensor}_mean_5"] = (
            features.groupby("engine_id")[sensor]
            .transform(
                lambda x: x.rolling(
                    window=5,
                    min_periods=1
                ).mean()
            )
        )

        # 5-cycle rolling standard deviation
        features[f"{sensor}_std_5"] = (
            features.groupby("engine_id")[sensor]
            .transform(
                lambda x: x.rolling(
                    window=5,
                    min_periods=1
                ).std()
            )
        )

    # Replace missing values
    features = features.fillna(0)

    return features


# ============================================================
# DETERMINE ENGINE HEALTH
# ============================================================

def get_health_status(rul):

    if rul > 80:
        return "HEALTHY"

    elif rul > 30:
        return "WARNING"

    else:
        return "CRITICAL"


# ============================================================
# SAVE TELEMETRY HISTORY LOCALLY
# ============================================================

def save_telemetry_history(history_records):

    TELEMETRY_HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    history_df = pd.DataFrame(history_records)

    history_df.to_csv(
        TELEMETRY_HISTORY_PATH,
        index=False
    )

    print()
    print("Telemetry history saved locally.")
    print(f"File: {TELEMETRY_HISTORY_PATH}")


# ============================================================
# UPLOAD TELEMETRY HISTORY TO AZURE
# ============================================================

def upload_to_azure():

    print()
    print("=" * 90)
    print("Uploading telemetry history to Azure...")
    print("=" * 90)

    try:

        # Authenticate using Azure CLI credentials
        credential = DefaultAzureCredential()

        # Azure Storage account URL
        account_url = (
            f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
        )

        # Create Blob Service Client
        blob_service_client = BlobServiceClient(
            account_url=account_url,
            credential=credential
        )

        # Get telemetry container
        container_client = (
            blob_service_client
            .get_container_client(CONTAINER_NAME)
        )

        # Open telemetry history
        with open(
            TELEMETRY_HISTORY_PATH,
            "rb"
        ) as data:

            container_client.upload_blob(
                name=BLOB_NAME,
                data=data,
                overwrite=True
            )

        print("Azure upload successful!")

        print(
            f"Blob: "
            f"{CONTAINER_NAME}/{BLOB_NAME}"
        )

    except Exception as e:

        print("Azure upload failed.")
        print(f"Error: {e}")


# ============================================================
# REPLAY ENGINE TELEMETRY
# ============================================================

def replay_engine(
    telemetry,
    engine_id,
    model,
    feature_columns,
    delay=0.1
):

    # Select one engine
    engine_data = (
        telemetry[
            telemetry["engine_id"] == engine_id
        ]
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    if engine_data.empty:

        print(f"Engine {engine_id} not found.")

        return None

    # Create the Digital Twin
    twin = DigitalTwin(engine_id)

    print()
    print("=" * 90)
    print(f"Digital Twin started | Engine {engine_id}")
    print(f"Total cycles: {len(engine_data)}")
    print("=" * 90)

    # Store telemetry history
    history = []

    # Store records for cloud/local storage
    telemetry_records = []

    # ========================================================
    # REAL-TIME TELEMETRY REPLAY
    # ========================================================

    for _, row in engine_data.iterrows():

        # ----------------------------------------------------
        # 1. Receive new telemetry
        # ----------------------------------------------------

        history.append(row.to_dict())

        history_df = pd.DataFrame(history)

        # ----------------------------------------------------
        # 2. Create rolling features
        # ----------------------------------------------------

        feature_data = create_features(history_df)

        # Current cycle only
        current_features = feature_data.iloc[[-1]]

        # Select exact model features
        X_current = current_features[feature_columns]

        # ----------------------------------------------------
        # 3. Predict Remaining Useful Life
        # ----------------------------------------------------

        predicted_rul = model.predict(X_current)[0]

        # RUL cannot be negative
        predicted_rul = max(0, predicted_rul)

        # ----------------------------------------------------
        # 4. Determine health state
        # ----------------------------------------------------

        health = get_health_status(predicted_rul)

        # ----------------------------------------------------
        # 5. Collect current sensor state
        # ----------------------------------------------------

        sensor_data = {
            sensor: float(row[sensor])
            for sensor in model_sensors
        }

        # ----------------------------------------------------
        # 6. Update Digital Twin
        # ----------------------------------------------------

        twin.update(
            cycle=int(row["cycle"]),
            sensor_data=sensor_data,
            predicted_rul=float(predicted_rul),
            health_status=health
        )

        # ----------------------------------------------------
        # 7. Store telemetry record
        # ----------------------------------------------------

        telemetry_record = {
            "engine_id": int(row["engine_id"]),
            "cycle": int(row["cycle"])
        }

        # Add sensor values
        telemetry_record.update(sensor_data)

        # Add Digital Twin prediction
        telemetry_record["predicted_rul"] = float(
            predicted_rul
        )

        telemetry_record["health_status"] = health

        telemetry_records.append(
            telemetry_record
        )

        # ----------------------------------------------------
        # 8. Display current Digital Twin state
        # ----------------------------------------------------

        print(
            f"Cycle: {int(row['cycle']):3d} | "
            f"RUL: {predicted_rul:7.2f} cycles | "
            f"Health: {health}"
        )

        # Simulate real-time delay
        time.sleep(delay)

    # ========================================================
    # FINAL DIGITAL TWIN STATE
    # ========================================================

    print("=" * 90)
    print("Digital Twin replay completed")
    print("=" * 90)

    print("\nFinal Digital Twin State:")

    state = twin.get_state()

    print(state)

    # ========================================================
    # SAVE TELEMETRY HISTORY
    # ========================================================

    save_telemetry_history(
        telemetry_records
    )

    # ========================================================
    # UPLOAD TO AZURE
    # ========================================================

    upload_to_azure()

    return twin


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("Loading NASA telemetry...")

    telemetry = load_telemetry()

    print(f"Records: {len(telemetry)}")
    print(f"Engines: {telemetry['engine_id'].nunique()}")

    print()
    print("Loading trained HistGradientBoosting model...")

    model, feature_columns = load_model()

    print("Model loaded successfully!")
    print(f"Model features: {len(feature_columns)}")

    # --------------------------------------------------------
    # Engine selected for simulation
    # --------------------------------------------------------

    engine_id = 49

    # --------------------------------------------------------
    # Start Digital Twin replay
    # --------------------------------------------------------

    replay_engine(
        telemetry=telemetry,
        engine_id=engine_id,
        model=model,
        feature_columns=feature_columns,
        delay=0.1
    )