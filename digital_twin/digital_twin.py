import csv
import json
from datetime import datetime
from pathlib import Path


class DigitalTwin:
    """
    Virtual representation of a turbofan engine.

    Maintains the latest engine state and records
    sensor telemetry, RUL predictions, and health
    evolution over time.
    """

    def __init__(self, engine_id):

        self.engine_id = engine_id
        self.current_cycle = None
        self.latest_sensors = {}
        self.predicted_rul = None
        self.health_status = "UNKNOWN"
        self.last_updated = None

        # ----------------------------------------------------
        # Storage paths
        # ----------------------------------------------------

        project_root = Path(__file__).resolve().parent.parent

        self.storage_dir = (
            project_root / "data" / "twin_state"
        )

        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.state_path = (
            self.storage_dir
            / f"engine_{engine_id}_state.json"
        )

        self.history_path = (
            self.storage_dir
            / "telemetry_history.csv"
        )

    # ========================================================
    # UPDATE DIGITAL TWIN
    # ========================================================

    def update(
        self,
        cycle,
        sensor_data,
        predicted_rul,
        health_status
    ):
        """Update the current Digital Twin state."""

        self.current_cycle = cycle
        self.latest_sensors = sensor_data
        self.predicted_rul = predicted_rul
        self.health_status = health_status
        self.last_updated = datetime.now()

        self.save_state()
        self.save_history()

    # ========================================================
    # GET CURRENT STATE
    # ========================================================

    def get_state(self):
        """Return the current Digital Twin state."""

        return {
            "engine_id": self.engine_id,
            "cycle": self.current_cycle,
            "predicted_rul": self.predicted_rul,
            "health_status": self.health_status,
            "last_updated": (
                self.last_updated.isoformat()
                if self.last_updated
                else None
            ),
            "sensor_count": len(self.latest_sensors)
        }

    # ========================================================
    # SAVE CURRENT STATE
    # ========================================================

    def save_state(self):
        """Save the latest Digital Twin state as JSON."""

        state = self.get_state()

        with open(
            self.state_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                state,
                file,
                indent=4
            )

    # ========================================================
    # SAVE TELEMETRY HISTORY
    # ========================================================

    def save_history(self):
        """Append telemetry and Digital Twin state to CSV."""

        row = {
            "engine_id": self.engine_id,
            "cycle": self.current_cycle,
            **self.latest_sensors,
            "predicted_rul": self.predicted_rul,
            "health_status": self.health_status,
            "timestamp": self.last_updated.isoformat()
        }

        # If the file does not exist or is empty,
        # create the header.
        file_exists = self.history_path.exists()
        file_empty = (
            not file_exists
            or self.history_path.stat().st_size == 0
        )

        with open(
            self.history_path,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=row.keys()
            )

            if file_empty:
                writer.writeheader()

            writer.writerow(row)