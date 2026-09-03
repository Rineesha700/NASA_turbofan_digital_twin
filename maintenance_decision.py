from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "twin_state"
    / "fleet_anomaly_history.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "twin_state"
    / "fleet_maintenance_history.csv"
)


# ---------------------------------------------------------
# MAINTENANCE DECISION
# ---------------------------------------------------------

def get_maintenance_decision(rul, anomaly_status):

    if rul <= 30:
        return "MAINTENANCE REQUIRED"

    elif rul <= 80 or anomaly_status == "ANOMALY":
        return "INSPECTION RECOMMENDED"

    else:
        return "MONITOR"


def get_reason(rul, anomaly_status):

    if rul <= 30 and anomaly_status == "ANOMALY":
        return "Critical RUL and abnormal sensor behavior"

    elif rul <= 30:
        return "Critical predicted RUL"

    elif anomaly_status == "ANOMALY":
        return "Abnormal sensor behavior detected"

    elif rul <= 80:
        return "Reduced predicted RUL"

    else:
        return "Normal operating condition"


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 80)
    print("MAINTENANCE DECISION ENGINE")
    print("=" * 80)

    print("\nLoading anomaly results...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Records: {len(df)}")
    print(f"Engines: {df['engine_id'].nunique()}")

    # -----------------------------------------------------
    # APPLY MAINTENANCE LOGIC
    # -----------------------------------------------------

    df["maintenance_decision"] = df.apply(
        lambda row: get_maintenance_decision(
            row["predicted_rul"],
            row["anomaly_status"]
        ),
        axis=1
    )

    df["maintenance_reason"] = df.apply(
        lambda row: get_reason(
            row["predicted_rul"],
            row["anomaly_status"]
        ),
        axis=1
    )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nMaintenance results saved:")
    print(OUTPUT_PATH)

    # -----------------------------------------------------
    # LATEST STATE FOR EACH ENGINE
    # -----------------------------------------------------

    latest_states = (
        df
        .sort_values(["engine_id", "cycle"])
        .groupby("engine_id")
        .tail(1)
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("MAINTENANCE DECISION SUMMARY")
    print("=" * 80)

    print(
        latest_states[
            "maintenance_decision"
        ]
        .value_counts()
        .to_string()
    )

    # -----------------------------------------------------
    # ENGINE SUMMARY
    # -----------------------------------------------------

    print("\nLatest Fleet Maintenance Status:")

    summary = latest_states[
        [
            "engine_id",
            "cycle",
            "predicted_rul",
            "health_status",
            "anomaly_status",
            "maintenance_decision",
            "maintenance_reason"
        ]
    ].copy()

    summary["predicted_rul"] = (
        summary["predicted_rul"]
        .round(1)
    )

    print(
        summary
        .sort_values("predicted_rul")
        .head(15)
        .to_string(index=False)
    )