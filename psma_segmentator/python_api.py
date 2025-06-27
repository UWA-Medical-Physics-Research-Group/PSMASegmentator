import os
from pathlib import Path
import csv
import torch
from importlib.metadata import version
import requests
import shutil
from psma_segmentator.download_weights import download_fold_weights_via_api
from psma_segmentator.inference import segmentate
from psma_segmentator.pre_processing import pre_process, shorten_path
from psma_segmentator.post_processing import post_process

def get_version_data(repo, version, headers):
    """
    Fetch release data from GitHub.

    Args:
        repo (str): GitHub repo name.
        version (str): Specific version to fetch. If None, fetch latest release.

    Returns:
        version (str): Version string.
        release_data (dict): Release metadata.
    """
    if version is None:
        api_url = f"https://api.github.com/repos/{repo}/releases/latest"
        print(f"\nFetching latest release from {repo}...")
    else:
        api_url = f"https://api.github.com/repos/{repo}/releases/tags/v{version}"
        print(f"\nFetching release {version} from {repo}...")

    response = requests.get(api_url, headers=headers)
    response.raise_for_status()
    release_data = response.json()
    selected_version = release_data["tag_name"].lstrip("v")

    return selected_version, release_data

def get_psmasegmentator_dir(version):
    """
    Get the path to the psmasegmentator directory, containing the model weights
    that have been downloaded.
    """
    
    if "PSMA_SEGMENTATOR_HOME" in os.environ:
        base_dir = Path(os.environ["PSMA_SEGMENTATOR_HOME"])
    else:
        base_dir = Path.home() / ".psmasegmentator"
    return base_dir / version

def setup_psma_segmentator(weights_dir: str):
    """
    Sets up the configuration for the PSMA Segmentator.
    Defines the directory structure and environment variables needed
    for the model weights and results.
    """
    os.environ["nnUNet_raw"] = str(weights_dir)
    os.environ["nnUNet_preprocessed"] = str(weights_dir)
    os.environ["nnUNet_results"] = str(weights_dir)

def psma_segmentator(weights_dir: str = None, 
                        input_dir: str = None,
                        output_dir: str = None, 
                        token: str = None,
                        version: str = None,
                        device: str = "cuda" if torch.cuda.is_available() else "cpu",
                        incl_rtstructs: bool = False,
                        verbose: bool = False,
                        overwrite: bool = False,
                        preprocess_only: bool = False,
                        postprocess_only: bool = False,
                        suv_thresh: float = 0.0,
                        organ_dir: str = None,
                        fast: bool = False
                    ):
    """
    Runs the PSMA segmentation pipeline, comprising pre-processing, segmentation, and post-processing.
    
    Args:
        weights_dir (str): Directory containing trained model weights. 
        input_dir (str): Directory containing PSMA PET/CT files.
        output_dir (str): Directory to save the segmentation results.
        token (str): API token for downloading weights.
        version (str): Version of the PSMASegmentator release weights to use. If None, uses the latest version.
        device (str): Device for inference ("cpu" or "cuda").
        incl_rtstructs (bool): Whether to include RTSTRUCT processing.
        verbose (bool): Verbosity level.
        overwrite (bool): Whether to overwrite existing files.
        preprocess_only (bool): If True, only pre-process the input files without segmentation.
        postprocess_only (bool): If True, only post-process the segmentation results.
        suv_thresh (float): SUV threshold for post-processing.
        organ_dir (str): Directory containing organ segmentations for post-processing lesion classification.
        fast (bool): If True, uses fast mode for inference, disabling TTA and using fast organ segmentation.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
    input_path = Path(input_dir)
    
    if incl_rtstructs == True and shutil.which("plastimatch") is None:
        raise EnvironmentError(
            "Plastimatch not found. Please install Plastimatch (e.g., via 'sudo apt install plastimatch') if you want to include RTSTRUCT processing."
        )

    if output_dir is None:
        output_dir = str(input_path.parent / f"{input_path.name}_outputs")
        print(f"\nOutput directory not specified. Using: {output_dir}")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir) and not preprocess_only:
        os.makedirs(output_dir)

    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "PSMASegmentator"
    }

    try:
        version, release_data = get_version_data(
                            repo="UWA-Medical-Physics-Research-Group/PSMASegmentator",
                            version=version, # if None, fetch latest, else fetch specified version
                            headers=headers)
        print(f"\nUsing PSMASegmentator version: {version}")
    except Exception as e:
        print(f"\nERROR: Could not fetch latest release from GitHub. "
            f"Check internet connection or repo access.\nError: {e}")
        raise SystemExit(1)  # hard exit

    if weights_dir is None:
        weights_dir = get_psmasegmentator_dir(version)
        print(f"\nUsing weights directory: {weights_dir}")
    
    setup_psma_segmentator(weights_dir)

    download_fold_weights_via_api(weights_dir, headers, release_data)  # Download model weights if needed

    if any(f.suffix == ".dcm" for f in input_path.rglob("*")):
        print(f"Input path {shorten_path(input_path)} contains DICOM files.")
        handling_dicom = True
        output_prepro_dir = str(input_path.parent / f"{input_path.name}_preprocessed")
        os.makedirs(output_prepro_dir, exist_ok=True)
    else:
        print(f"Input path {shorten_path(input_path)} contains NIfTI files.")
        handling_dicom = False
        output_prepro_dir = str(input_path)
    
    if not postprocess_only:
        # Preprocess the input files
        list_of_lists = pre_process(input_path, incl_rtstructs, 
                                    output_pred_dir=output_dir,
                                    output_prepro_dir=output_prepro_dir,
                                    handling_dicom=handling_dicom,
                                    verbose=verbose, overwrite=overwrite)
        if preprocess_only:
            print("\nPre-processing (only) complete. No segmentation performed.")
            return
                
        segmentate(
            model_folder=weights_dir,
            list_of_lists=list_of_lists,
            output_dir=output_dir,
            device=device,
            use_tta=not fast,  # Use TTA unless fast mode is specified 
            verbose=verbose
        )
    else:
        print("\nSkipping pre-processing and segmentation. Only post-processing will be performed.")
        # Check if output dir is empty (i.e., nothing to post-process)
        if not os.listdir(output_dir):
            print(f"\nERROR: Output directory {output_dir} is empty. "
                "Please run pre-processing and segmentation first.")
            raise SystemExit(1)

    # Post-process the segmentation results
    print(f"\nInitiating post-processing of segmentations in {shorten_path(output_dir)}...")
    post_process(
        prepro_dir=output_prepro_dir,
        output_dir=output_dir,
        organ_dir=organ_dir,
        device=device,
        suv_thresh=suv_thresh,
        fast=fast,
        verbose=verbose,
        overwrite=overwrite
    )

    print("\nPSMA segmentation pipeline complete.")
    return