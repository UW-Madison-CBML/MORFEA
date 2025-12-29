import argparse
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# 1. Configuration
# REQUIRED: Set the folder ID of a folder shared with your service account
#
# HOW TO SET THIS UP:
# 1. Create a folder in your Google Drive
# 2. Right-click the folder -> Share -> Add the service account email:
#    jens-serv@chicago-transit-379219.iam.gserviceaccount.com
#    (Give it "Editor" permission)
# 3. Open the folder in Drive, copy the ID from the URL:
#    https://drive.google.com/drive/folders/FOLDER_ID_HERE
# 4. Paste the ID below:
FOLDER_ID = os.environ.get('GOOGLE_DRIVE_FOLDER_ID', None)

def upload_file(file_path):
    # Ensure the file exists locally
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # Check if folder ID is set
    if not FOLDER_ID:
        print("\n" + "="*70)
        print("ERROR: Google Drive Folder ID not set!")
        print("="*70)
        print("\nService accounts cannot upload to 'My Drive' (no storage quota).")
        print("You must upload to a shared folder instead.\n")
        print("SETUP INSTRUCTIONS:")
        print("1. Go to Google Drive and create a folder for uploads")
        print("2. Right-click the folder -> Share")
        print("3. Add this email with 'Editor' permission:")
        print("   jens-serv@chicago-transit-379219.iam.gserviceaccount.com")
        print("4. Open the folder, copy the ID from the URL:")
        print("   https://drive.google.com/drive/folders/YOUR_FOLDER_ID_HERE")
        print("5. Set the environment variable:")
        print("   export GOOGLE_DRIVE_FOLDER_ID='your_folder_id_here'")
        print("="*70 + "\n")
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
        print(f"Target folder ID: {FOLDER_ID}")

        # Verify we can access the folder
        try:
            folder_info = service.files().get(
                fileId=FOLDER_ID,
                fields='id, name, capabilities',
                supportsAllDrives=True
            ).execute()
            print(f"Folder found: '{folder_info.get('name')}'")

            # Check if we can edit
            caps = folder_info.get('capabilities', {})
            if not caps.get('canAddChildren', False):
                print("\n⚠ WARNING: Service account cannot add files to this folder!")
                print("Make sure the folder is shared with 'Editor' permission, not 'Viewer'")
                return
        except Exception as e:
            print(f"\n✗ Cannot access folder {FOLDER_ID}")
            print(f"Error: {e}")
            print("\nMake sure:")
            print("1. The folder ID is correct")
            print("2. The folder is shared with: jens-serv@chicago-transit-379219.iam.gserviceaccount.com")
            print("3. The service account has 'Editor' permission")
            return
    else:
        print("ERROR: No folder ID - this shouldn't happen (should have been caught earlier)")
        return

    # 4. Perform Upload
    print(f"Uploading '{file_name}' to Google Drive folder...")
    media = MediaFileUpload(file_path, resumable=True)

    try:
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, webViewLink',
            supportsAllDrives=True  # Enable support for Shared Drives
        ).execute()

        print(f"\n✓ Successfully uploaded!")
        print(f"  File Name: {file.get('name')}")
        print(f"  File ID: {file.get('id')}")
        print(f"  View Link: {file.get('webViewLink')}")

    except Exception as e:
        print(f"\n✗ Upload failed!")
        print(f"Error: {e}")
        if 'storageQuotaExceeded' in str(e):
            print("\nThis error means the folder isn't properly shared with the service account.")
            print("Make sure you shared the folder with:")
            print("  jens-serv@chicago-transit-379219.iam.gserviceaccount.com")
        elif '404' in str(e):
            print("\nFolder not found. Check that FOLDER_ID is correct.")
        elif '403' in str(e):
            print("\nPermission denied. Make sure the service account has 'Editor' access.")

if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Upload a file to Google Drive.')
    parser.add_argument('filename', help='The path to the file you want to upload')
    
    args = parser.parse_args()
    upload_file(args.filename)
