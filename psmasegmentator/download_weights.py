import os
import requests
import zipfile
from pathlib import Path

def download_fold_weights(github_base_url, output_dir, fold_numbers = [0,1,2,3,4], cleanup=True):
    """
    Downloads and extracts pre-trained weights for individual folds from GitHub release assets.

    Args:
        github_base_url (str): The base URL for the GitHub assets (e.g., "https://github.com/user/repo/releases/download/v1.0").
        output_dir (str): The directory where the extracted files should be saved.
        fold_numbers (list): List of fold numbers to download (e.g., [0, 1, 2, 3, 4]).
        cleanup (bool): Whether to delete the downloaded zip files after extraction. Defaults to True.

    Returns:
        str: Path to the directory containing the complete folder structure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download dataset.json
    dataset_json_path = output_dir / "dataset.json"
    if not dataset_json_path.exists():
        dataset_json_url = f"{github_base_url}/dataset.json"
        print(f"Downloading dataset.json from {dataset_json_url}...")
        response = requests.get(dataset_json_url)
        response.raise_for_status()
        with open(dataset_json_path, "wb") as json_file:
            json_file.write(response.content)
        print(f"Downloaded dataset.json to {dataset_json_path}")

    # Download plans.json
    plans_json_path = output_dir / "plans.json"
    if not plans_json_path.exists():
        plans_json_url = f"{github_base_url}/plans.json"
        print(f"Downloading plans.json from {plans_json_url}...")
        response = requests.get(plans_json_url)
        response.raise_for_status()
        with open(plans_json_path, "wb") as json_file:
            json_file.write(response.content)
        print(f"Downloaded plans.json to {plans_json_path}")

    # Download and extract each fold
    for fold in fold_numbers:
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        zip_file_path = fold_dir / f"fold_{fold}.zip"
        fold_url = f"{github_base_url}/fold_{fold}.zip"

        try:
            print(f"Downloading fold {fold} weights from {fold_url}...")
            response = requests.get(fold_url, stream=True)
            response.raise_for_status()

            # Save the downloaded file
            with open(zip_file_path, "wb") as zip_file:
                for chunk in response.iter_content(chunk_size=8192):
                    zip_file.write(chunk)
            print(f"Downloaded fold {fold} weights to {zip_file_path}")

            # Extract the zip file
            print(f"Extracting fold {fold} weights to {fold_dir}...")
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(fold_dir)
            print(f"Extraction complete for fold {fold}.")

            # Cleanup
            if cleanup:
                os.remove(zip_file_path)
                print(f"Removed temporary zip file for fold {fold}: {zip_file_path}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading weights for fold {fold}: {e}")
            raise

        except zipfile.BadZipFile:
            print(f"The downloaded file for fold {fold} is not a valid zip archive.")
            raise

    return str(output_dir)
