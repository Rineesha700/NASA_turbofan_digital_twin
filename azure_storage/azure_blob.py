from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from pathlib import Path


# Azure Storage Account
STORAGE_ACCOUNT_NAME = "nasatwindata2026"
CONTAINER_NAME = "telemetry"

# Local telemetry file
LOCAL_FILE = Path("data/twin_state/telemetry_history.csv")

# Azure Blob name
BLOB_NAME = "telemetry_history.csv"


def upload_telemetry():
    """Upload telemetry history CSV to Azure Blob Storage."""

    print("Connecting to Azure Storage...")

    # Authenticate using Azure CLI login
    credential = DefaultAzureCredential()

    # Create Blob Service Client
    account_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"

    blob_service_client = BlobServiceClient(
        account_url=account_url,
        credential=credential
    )

    # Get the telemetry container
    container_client = blob_service_client.get_container_client(
        CONTAINER_NAME
    )

    # Check local file
    if not LOCAL_FILE.exists():
        print(f"ERROR: File not found: {LOCAL_FILE}")
        return

    # Upload file
    print(f"Uploading: {LOCAL_FILE}")
    print(f"Container: {CONTAINER_NAME}")
    print(f"Blob: {BLOB_NAME}")

    with open(LOCAL_FILE, "rb") as data:
        container_client.upload_blob(
            name=BLOB_NAME,
            data=data,
            overwrite=True
        )

    print("Upload successful!")
    print(
        f"Azure Blob: "
        f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net/"
        f"{CONTAINER_NAME}/{BLOB_NAME}"
    )


if __name__ == "__main__":
    upload_telemetry()