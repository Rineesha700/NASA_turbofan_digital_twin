from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "twin_state"
    / "fleet_maintenance_history.csv"
)

# Azure configuration
STORAGE_ACCOUNT = "nasatwindata2026"
CONTAINER_NAME = "telemetry"
BLOB_NAME = "fleet_maintenance_history.csv"

# Connect to Azure
account_url = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net"
credential = DefaultAzureCredential()

blob_service_client = BlobServiceClient(
    account_url=account_url,
    credential=credential
)

# Check local file
if not LOCAL_FILE.exists():
    raise FileNotFoundError(
        f"Fleet data not found: {LOCAL_FILE}"
    )

# Upload
blob_client = blob_service_client.get_blob_client(
    container=CONTAINER_NAME,
    blob=BLOB_NAME
)

with open(LOCAL_FILE, "rb") as data:
    blob_client.upload_blob(data, overwrite=True)

print("Fleet data uploaded successfully!")
print(f"Azure Blob: {BLOB_NAME}")
print(f"URL: {blob_client.url}")