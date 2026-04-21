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

NB: This software is intended for RESEARCH PURPOSES ONLY.
"""

import os
import re
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm


def _extract_fold_zip(fold_zip_path: Path, target_dir: Path, cleanup: bool = True):
    """Extract a fold_*.zip and normalize nested fold directories if needed."""
    fold_match = re.match(r"^fold_(\d+)\.zip$", fold_zip_path.name)
    fold_number = fold_match.group(1) if fold_match else None

    with zipfile.ZipFile(fold_zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    if fold_number is not None:
        fold_dir = target_dir / f"fold_{fold_number}"
        nested_dir = fold_dir / f"fold_{fold_number}"
        if nested_dir.exists() and nested_dir.is_dir():
            for file in nested_dir.iterdir():
                file.rename(fold_dir / file.name)
            nested_dir.rmdir()

    if cleanup and fold_zip_path.exists():
        fold_zip_path.unlink()


def prepare_standard_model_weights(weights_dir,
                                    checkpoint_name: str = "checkpoint_final.pth",
                                    cleanup: bool = True):
    """
    Resolve and prepare standard model weights from a weights directory.

    Supports:
    1) Already extracted fold_* directories with checkpoints
    2) Top-level fold_*.zip files that need extraction
    """
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Only consider top-level nnUNet fold directories for standard model detection.
    # Do NOT use recursive discovery here because nested fast model folders may also
    # contain checkpoint files and would create false positives.
    existing_valid_folds = [
        p for p in weights_dir.glob("fold_*")
        if p.is_dir() and (p / checkpoint_name).exists()
    ]
    if len(existing_valid_folds) > 0:
        return str(weights_dir.resolve())

    # Extract any local fold zip archives.
    fold_zip_files = sorted(weights_dir.glob("fold_*.zip"))
    for fold_zip in fold_zip_files:
        fold_match = re.match(r"^fold_(\d+)\.zip$", fold_zip.name)
        if fold_match:
            fold_n = fold_match.group(1)
            fold_checkpoint = weights_dir / f"fold_{fold_n}" / checkpoint_name
            if fold_checkpoint.exists():
                continue
        print(f"Extracting standard fold archive {fold_zip.name}...")
        _extract_fold_zip(fold_zip, weights_dir, cleanup=cleanup)

    existing_valid_folds = [
        p for p in weights_dir.glob("fold_*")
        if p.is_dir() and (p / checkpoint_name).exists()
    ]
    if len(existing_valid_folds) == 0:
        raise FileNotFoundError(
            f"No standard model checkpoints ({checkpoint_name}) found under {weights_dir}."
        )

    return str(weights_dir.resolve())


def prepare_fast_model_weights(weights_dir,
                               headers=None,
                               release_data=None,
                               cleanup=True):
    """
    Resolve and prepare the fast-model directory from a weights directory.

    This supports three cases:
    1) Already-extracted fast directory exists (fast_v*/fold_*/checkpoint*.pth)
    2) Local fast_v*.zip exists in weights_dir and must be extracted
    3) fast_v*.zip must be downloaded from release assets (if release_data + headers provided)
    """
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    def _has_top_level_fast_folds(candidate_dir: Path) -> bool:
        return any(
            p.is_dir() and (p / "checkpoint_final.pth").exists()
            for p in candidate_dir.glob("fold_*")
        )

    def _prepare_candidate(candidate_dir: Path) -> bool:
        if not candidate_dir.exists() or not candidate_dir.is_dir():
            return False
        # Ensure any fold zip archives in this candidate are extracted.
        for fold_zip in sorted(candidate_dir.glob("fold_*.zip")):
            fold_match = re.match(r"^fold_(\d+)\.zip$", fold_zip.name)
            if fold_match:
                fold_n = fold_match.group(1)
                fold_checkpoint = candidate_dir / f"fold_{fold_n}" / "checkpoint_final.pth"
                if fold_checkpoint.exists():
                    continue
            print(f"Extracting fast fold archive {fold_zip.name}...")
            _extract_fold_zip(fold_zip, candidate_dir, cleanup=cleanup)
        return _has_top_level_fast_folds(candidate_dir)

    # Try already-existing candidates first.
    initial_candidates = []
    if weights_dir.name.startswith("fast_v"):
        initial_candidates.append(weights_dir)
    initial_candidates.extend(sorted([p for p in weights_dir.glob("fast_v*") if p.is_dir()]))

    seen = set()
    for candidate in initial_candidates:
        c = candidate.resolve()
        if c in seen:
            continue
        seen.add(c)
        if _prepare_candidate(candidate):
            return str(candidate.resolve())

    # If not found, try local fast zip.
    fast_zip_files = sorted(weights_dir.glob("fast_v*.zip"))
    fast_zip_path = fast_zip_files[0] if fast_zip_files else None

    # If still missing, optionally download from GitHub release assets.
    if fast_zip_path is None and release_data is not None and headers is not None:
        fast_asset = next(
            (a for a in release_data["assets"] if a["name"].startswith("fast_v") and a["name"].endswith(".zip")),
            None,
        )
        if fast_asset is not None:
            fast_zip_path = weights_dir / fast_asset["name"]
            if not fast_zip_path.exists():
                print(f"Downloading {fast_asset['name']}...")
                download_file_from_api(fast_asset["url"], fast_zip_path, headers, progress_bar=True)
                print(f"Downloaded fast model archive to {fast_zip_path}")

    if fast_zip_path is None:
        raise FileNotFoundError(
            f"Fast model weights not found in {weights_dir}. Expected fast_v*.zip or extracted fast_v* directory."
        )

    # Avoid creating fast_vX/fast_vX when the provided weights_dir is already fast_vX.
    if weights_dir.name.startswith("fast_v") and fast_zip_path.stem == weights_dir.name:
        extraction_root = weights_dir
    else:
        extraction_root = weights_dir / fast_zip_path.stem
        extraction_root.mkdir(parents=True, exist_ok=True)

    print(f"Extracting fast model archive: {fast_zip_path.name} to {extraction_root}...")
    with zipfile.ZipFile(fast_zip_path, "r") as zip_ref:
        zip_ref.extractall(extraction_root)
    if cleanup and fast_zip_path.exists():
        fast_zip_path.unlink()

    # Normalize accidental double nesting: fast_vX/fast_vX/... -> fast_vX/...
    # This happens when the archive already contains a top-level fast_vX folder and
    # we extract into a folder with the same name.
    nested_same_name_dir = extraction_root / fast_zip_path.stem
    if (
        nested_same_name_dir.exists()
        and nested_same_name_dir.is_dir()
        and extraction_root.name == fast_zip_path.stem
    ):
        has_top_level_fast_content = any(extraction_root.glob("fold_*")) or any(extraction_root.glob("fold_*.zip"))
        has_nested_fast_content = any(nested_same_name_dir.glob("fold_*")) or any(nested_same_name_dir.glob("fold_*.zip"))
        # Only flatten when safe and clearly redundant.
        if has_nested_fast_content and not has_top_level_fast_content:
            for item in nested_same_name_dir.iterdir():
                target = extraction_root / item.name
                if target.exists():
                    continue
                item.rename(target)
            if not any(nested_same_name_dir.iterdir()):
                nested_same_name_dir.rmdir()

    # Re-scan candidates after extraction and return first valid model folder.
    post_extract_candidates = [extraction_root]
    post_extract_candidates.extend(sorted([p for p in extraction_root.glob("fast_v*") if p.is_dir()]))
    if weights_dir != extraction_root:
        post_extract_candidates.extend(sorted([p for p in weights_dir.glob("fast_v*") if p.is_dir()]))
    if weights_dir.name.startswith("fast_v"):
        post_extract_candidates.append(weights_dir)

    seen = set()
    for candidate in post_extract_candidates:
        c = candidate.resolve()
        if c in seen:
            continue
        seen.add(c)
        if _prepare_candidate(candidate):
            return str(candidate.resolve())

    raise FileNotFoundError(
        f"No fast model checkpoints found under {weights_dir} after preparing fast weights."
    )

def download_model_weights_via_api(output_dir, 
                                    headers, 
                                    release_data,
                                    fold_numbers=[0, 1, 2, 3, 4], 
                                    cleanup=True,
                                    fast=False,
                                    checkpoint_name: str = "checkpoint_final.pth"):
    """
    Downloads and extracts pre-trained weights for the current software version from GitHub release assets.

    Args:
        output_dir (str): The directory where the extracted files should be saved.
        token (str): PAT for github repo
        release_data (dict): Release data containing asset information.
        fold_numbers (list): List of fold numbers to download (e.g., [0, 1, 2, 3, 4]).
        cleanup (bool): Whether to delete the downloaded zip files after extraction. Defaults to True.

    Returns:
        str: Path to the directory containing the model folder for inference.
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

    # If fast mode, skip standard folds and go directly to fast weights resolution.
    if not fast:
        # Download and extract each fold
        for fold in fold_numbers:
            fold_dir = output_dir / f"fold_{fold}"
            if fold_dir.exists() and (fold_dir / checkpoint_name).exists():
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
                _extract_fold_zip(zip_file_path, output_dir, cleanup=cleanup)

                print(f"Extraction complete for fold {fold}.")
                if cleanup:
                    print(f"Removed temporary zip file for fold {fold}: {zip_file_path}")

            except requests.exceptions.RequestException as e:
                print(f"Error downloading weights for fold {fold}: {e}")
                raise

            except zipfile.BadZipFile:
                print(f"The downloaded file for fold {fold} is not a valid zip archive.")
                raise


    if fast:
        return prepare_fast_model_weights(
            weights_dir=output_dir,
            headers=headers,
            release_data=release_data,
            cleanup=cleanup,
        )

    return prepare_standard_model_weights(
        weights_dir=output_dir,
        checkpoint_name=checkpoint_name,
        cleanup=cleanup,
    )

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
    # Ensure correct Accept header for ZIP files without mutating shared headers
    request_headers = dict(headers or {})
    request_headers["Accept"] = "application/octet-stream"

    response = requests.get(asset_url, headers=request_headers, stream=True)
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