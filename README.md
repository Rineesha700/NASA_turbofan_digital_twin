# NASA Turbofan Digital Twin

A machine-learning-based digital twin prototype for monitoring turbofan engine health and supporting predictive maintenance using NASA C-MAPSS data.

## Overview

The project replays turbofan engine telemetry and maintains a digital representation of engine state. Machine learning is used to estimate Remaining Useful Life (RUL), detect abnormal sensor behavior, and generate maintenance recommendations.

## Architecture

NASA C-MAPSS Data
        ↓
Telemetry Replay
        ↓
Azure Blob Storage
        ↓
Data Processing
        ↓
Digital Twin State
        ↓
RUL Prediction + Anomaly Detection
        ↓
Maintenance Decision
        ↓
Streamlit Dashboard

## Key Features

- Simulated telemetry replay for 100 turbofan engines
- RUL prediction using machine learning
- Anomaly detection using Isolation Forest
- Engine health classification: Healthy / Warning / Critical
- Explainable maintenance recommendations
- Azure Blob Storage integration
- Interactive Streamlit fleet monitoring dashboard
- Engine-level sensor and RUL visualization

## Tech Stack

Python • Pandas • NumPy • Scikit-learn • XGBoost • Azure Blob Storage • Streamlit • Plotly

## Project Structure

```text
NASA_turbofan_digital_twin/
├── azure_storage/       # Azure Blob Storage integration
├── dashboard/           # Streamlit dashboard
├── digital_twin/        # Digital Twin state management
├── models/              # Trained ML models
├── notebooks/           # Data exploration
├── simulator/           # Telemetry and fleet replay
├── anomaly_detection.py
├── maintenance_decision.py
├── requirements.txt
└── README.md
```
## Dataset

NASA C-MAPSS Turbofan Engine Degradation Simulation dataset.The raw NASA dataset is excluded from this repository and can be obtained from NASA.

Run the Dashboard
pip install -r requirements.txt
2. streamlit run dashboard/app.py

## Disclaimer

This is a portfolio prototype demonstrating Digital Twin, machine learning, anomaly detection, cloud storage, and predictive maintenance concepts using simulated telemetry derived from NASA C-MAPSS data.

## Output
<img width="1916" height="956" alt="image" src="https://github.com/user-attachments/assets/fb03e8f6-e5a3-4cc0-9b79-d324cf3d9e2f" />
<img width="1907" height="967" alt="image" src="https://github.com/user-attachments/assets/d5f9b1b7-be7d-4c04-a174-5b321f001e25" />
<img width="1906" height="952" alt="image" src="https://github.com/user-attachments/assets/6cf558bb-eba3-4bdf-a911-526ed1e30719" />
<img width="1917" height="926" alt="image" src="https://github.com/user-attachments/assets/2227c6d4-e5ee-4d8a-9cf7-5994c47fe749" />
<img width="1907" height="916" alt="image" src="https://github.com/user-attachments/assets/cdc82c18-96a2-497f-8760-955afa33ba02" />
<img width="1910" height="915" alt="image" src="https://github.com/user-attachments/assets/c8177b56-efe9-4276-8a75-7ce489197206" />





