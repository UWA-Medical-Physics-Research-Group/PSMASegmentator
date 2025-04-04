import os
from pathlib import Path
import csv
import torch
from importlib.metadata import version
import psma_segmentator.post_processing
from psma_segmentator.download_weights import download_fold_weights_via_api
from psma_segmentator.run_inference import nnUNet_predict_image

# __version__ = version("psma_segmentator")
__version__ = "0.0.2"

def get_psmasegmentator_dir(version=__version__):
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
    print(f"Setting up PSMA Segmentator Version {__version__}")
    os.environ["nnUNet_raw"] = str(weights_dir)
    os.environ["nnUNet_preprocessed"] = str(weights_dir)
    os.environ["nnUNet_results"] = str(weights_dir)

def psma_segmentator(weights_dir: str = None, 
                        input_dir: str = None,
                        token: str = None,
                        output_dir: str = None, 
                        # file_format: str = "dicom",
                        device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Runs the PSMA segmentation pipeline, including pre-processing and segmentation.
    
    Args:
        weights_dir (str): Directory containing trained model weights. 
        token (str): API token for downloading weights.
        input_dir (str): Directory containing PSMA PET/CT files.
        output_dir (str): Directory to save the segmentation results.
        device (str): Device for inference ("cpu" or "cuda").
        file_format (str): Input file format, either "dicom" or "nifti". Defaults to "dicom".
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

    if weights_dir is None:
        weights_dir = get_psmasegmentator_dir()
    print(f"Using weights directory: {weights_dir}")
    
    setup_psma_segmentator(weights_dir)

    download_fold_weights_via_api(weights_dir, token)  # Download model weights if needed
    
    # Collect unique case names (removing _0000.nii.gz and _0001.nii.gz suffixes)
    case_names = set()
    for filename in os.listdir(input_dir):
        if filename.endswith(".nii.gz"):
            case_name = filename.rsplit("_000", 1)[0]  # Remove _0000.nii.gz or _0001.nii.gz
            case_names.add(case_name)
    
    num_cases = len(case_names)
    print(f"Number of cases found: {num_cases}")
        
    # Process each case
    for i, case_name in enumerate(case_names, start=1):
        ct_dir = os.path.join(input_dir, f"{case_name}_0000.nii.gz")
        pet_dir = os.path.join(input_dir, f"{case_name}_0001.nii.gz")
        if not os.path.exists(ct_dir) or not os.path.exists(pet_dir):
            print(f"Warning: Missing files for case {case_name}. Skipping.")
            continue
        # Create a directory for each case's output
        save_dir = os.path.join(output_dir, case_name)
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Processing {case_name} ({i}/{num_cases}). Cases left: {num_cases - i}")
        
        nnUNet_predict_image(model_folder=weights_dir, 
                                ct_input=ct_dir, pet_input=pet_dir,
                                output_path=save_dir, 
                                # image_type=file_format, 
                                device=device, step_size=0.5, 
                                use_tta=True, verbose=False)
    
    print("PSMA segmentation pipeline complete.")

