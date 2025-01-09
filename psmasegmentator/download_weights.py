import os
import requests
import zipfile
from pathlib import Path

def download_weights(github_url, output_dir, cleanup=True):
    """
    Downloads and extracts pre-trained weights from a GitHub release asset.

    Args:
        github_url (str): The direct URL to the GitHub asset (zip file).
        output_dir (str): The directory where the extracted files should be saved.
        cleanup (bool): Whether to delete the downloaded zip file after extraction. Defaults to True.

    Returns:
        str: Path to the extracted weights directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Temporary path for the downloaded zip file
    zip_file_path = output_dir / "temp_weights.zip"

    try:
        print(f"Downloading weights from {github_url}...")
        response = requests.get(github_url, stream=True)
        response.raise_for_status()  # Raise an error for HTTP codes >= 400

        # Save the downloaded file
        with open(zip_file_path, "wb") as zip_file:
            for chunk in response.iter_content(chunk_size=8192):
                zip_file.write(chunk)
        print(f"Downloaded weights to {zip_file_path}")

        # Extract the zip file
        print(f"Extracting weights to {output_dir}...")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extraction complete.")

        # Cleanup
        if cleanup:
            os.remove(zip_file_path)
            print(f"Removed temporary zip file: {zip_file_path}")

        return str(output_dir)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading weights: {e}")
        raise

    except zipfile.BadZipFile:
        print(f"The downloaded file is not a valid zip archive.")
        raise