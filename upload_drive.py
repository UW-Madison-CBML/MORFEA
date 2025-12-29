import argparse
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# 1. Configuration
# Optional: Set a specific folder ID if you want to upload into a folder
# Otherwise, it uploads to the service account's root drive.
FOLDER_ID = None

def upload_file(file_path):
    # Ensure the file exists locally
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # 2. Authenticate using Service Account credentials
    # Get credentials path from environment variable
    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not creds_path:
        print("Error: GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
        return

    if not os.path.exists(creds_path):
        print(f"Error: Credentials file not found at '{creds_path}'")
        return

    # Load service account credentials with proper scopes
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    creds = service_account.Credentials.from_service_account_file(
        creds_path, scopes=SCOPES)

    # Build the Drive service
    service = build('drive', 'v3', credentials=creds)

    # 3. Prepare File Metadata
    file_name = os.path.basename(file_path)
    file_metadata = {'name': file_name}
    
    if FOLDER_ID:
        file_metadata['parents'] = [FOLDER_ID]

    # 4. Perform Upload
    print(f"Uploading {file_name}...")
    media = MediaFileUpload(file_path, resumable=True)
    
    try:
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        print(f"Successfully uploaded! File ID: {file.get('id')}")
    except Exception as e:
        print(f"An error occurred during upload: {e}")

if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Upload a file to Google Drive.')
    parser.add_argument('filename', help='The path to the file you want to upload')
    
    args = parser.parse_args()
    upload_file(args.filename)
