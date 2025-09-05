""" 
PSMASegmentator is a tool for the automatic segmentation of PSMA 
PET/CT images using Deep Learning (DL).
Copyright (C) 2025 UWA Medical Physics Research Group, The University 
of Western Australia, Crawley, WA, Australia

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import requests
import zipfile
import os
from pathlib import Path
from importlib.metadata import version

def download_model_weights_via_api(output_dir, 
                                    headers, 
                                    release_data,
                                    fold_numbers=[0, 1, 2, 3, 4], 
                                    cleanup=True):
    """
    Downloads and extracts pre-trained weights for the current software version from GitHub release assets.

    Args:
        output_dir (str): The directory where the extracted files should be saved.
        token (str): PAT for github repo
        release_data (dict): Release data containing asset information.
        fold_numbers (list): List of fold numbers to download (e.g., [0, 1, 2, 3, 4]).
        cleanup (bool): Whether to delete the downloaded zip files after extraction. Defaults to True.

    Returns:
        str: Path to the directory containing the complete folder structure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # repo = "UWA-Medical-Physics-Research-Group/PSMASegmentator"

    # headers = {
    #     "Authorization": f"Bearer {token}",
    #     "User-Agent": "PSMASegmentator"
    # }

    # # Get release data
    # api_url = f"https://api.github.com/repos/{repo}/releases/tags/{'v' + __version__}"
    # response = requests.get(api_url, headers=headers)
    # response.raise_for_status()
    # release_data = response.json()

    # Download dataset.json
    dataset_json_path = output_dir / "dataset.json"
    if not dataset_json_path.exists():
        asset = next((a for a in release_data["assets"] if a["name"] == "dataset.json"), None)
        if asset:
            print("Downloading dataset.json...")
            download_file_from_api(asset["url"], dataset_json_path, headers)
            print(f"Downloaded dataset.json to {dataset_json_path}")

    # Download plans.json
    plans_json_path = output_dir / "plans.json"
    if not plans_json_path.exists():
        asset = next((a for a in release_data["assets"] if a["name"] == "plans.json"), None)
        if asset:
            print("Downloading plans.json...")
            download_file_from_api(asset["url"], plans_json_path, headers)
            print(f"Downloaded plans.json to {plans_json_path}")

    # Download liver classifier model if present (version-agnostic, .pth extension)
    liver_asset = next((a for a in release_data["assets"] if "liver_classifier" in a["name"] and a["name"].endswith(".pth")), None)
    if liver_asset:
        liver_model_path = output_dir / liver_asset["name"]
        if not liver_model_path.exists():
            print(f"Downloading {liver_asset['name']}...")
            download_file_from_api(liver_asset["url"], liver_model_path, headers)
            print(f"Downloaded liver classifier model to {liver_model_path}")
        else:
            print(f"Liver classifier model already exists. Skipping download.")

    # Download and extract each fold
    for fold in fold_numbers:
        fold_dir = output_dir / f"fold_{fold}"
        if fold_dir.exists() and (fold_dir / "checkpoint_final.pth").exists():
            print(f"Fold {fold} already exists. Skipping download.")
            continue

        fold_asset_name = f"fold_{fold}.zip"
        asset = next((a for a in release_data["assets"] if a["name"] == fold_asset_name), None)

        if not asset:
            print(f"Skipping fold {fold}: Asset not found.")
            continue

        zip_file_path = output_dir / fold_asset_name
        try:
            print(f"Downloading {fold_asset_name}...")
            download_file_from_api(asset["url"], zip_file_path, headers, progress_bar=True)
            print(f"Downloaded fold {fold} weights to {zip_file_path}")

            # Extract the zip file
            print(f"Extracting fold {fold} weights to {fold_dir}...")
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)

            # Move the files out of nested directories if necessary
            nested_dir = fold_dir / f"fold_{fold}"
            if nested_dir.exists() and nested_dir.is_dir():
                for file in nested_dir.iterdir():
                    file.rename(fold_dir / file.name)
                nested_dir.rmdir()  # Remove the now-empty nested directory

            print(f"Extraction complete for fold {fold}.")

            # Cleanup
            if cleanup:
                zip_file_path.unlink()  # Remove the zip file
                print(f"Removed temporary zip file for fold {fold}: {zip_file_path}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading weights for fold {fold}: {e}")
            raise

        except zipfile.BadZipFile:
            print(f"The downloaded file for fold {fold} is not a valid zip archive.")
            raise

def download_file_from_api(asset_url, local_path, headers, progress_bar=True):
    """
    Downloads a file from a GitHub release asset URL with an optional progress bar.

    Args:
        asset_url (str): The GitHub API URL for the asset.
        local_path (Path): The local file path to save the asset.
        headers (dict): Headers for the request (e.g., authentication headers).
        progress_bar (bool): Whether to display a progress bar. Defaults to True.

    Returns:
        None
    """
    # Ensure correct Accept header for ZIP files
    headers["Accept"] = "application/octet-stream"

    response = requests.get(asset_url, headers=headers, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with open(local_path, "wb") as file, tqdm(
        desc=f"Downloading {local_path.name}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        disable=not progress_bar
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            bar.update(len(chunk))

    print(f"File downloaded to {local_path}")