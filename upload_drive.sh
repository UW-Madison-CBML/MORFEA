#!/bin/bash

# 2. Install required Google API libraries
echo "Installing Google API libraries..."
pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib

# 3. Set the Environment Variable for credentials
# REPLACE 'your-service-account-file.json' with your actual filename
export GOOGLE_DRIVE_FOLDER_ID="0AABJos17r17TUk9PVA"
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/googlecloud.json"
echo "GOOGLE_APPLICATION_CREDENTIALS set to: $GOOGLE_APPLICATION_CREDENTIALS"

# 4. Unzip the latents.tar.gz file
if [ -f "latents.tar.gz" ]; then
    echo "Extracting latents.tar.gz..."
    tar -xzvf latents.tar.gz
else
    echo "Error: latents.tar.gz not found in current directory."
fi

echo "Setup complete."

python upload_drive.py latents.tar.gz
