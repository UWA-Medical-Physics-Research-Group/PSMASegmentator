import sys
import os
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from psmasegmentator.download_weights import download_fold_weights_via_api
from psmasegmentator.inference import nnUNet_predict_image
from psmasegmentator.config import get_psmasegmentator_dir, setup_psmasegmentator


def psmasegmentator (pet_dir=None, 
                     ct_dir = None,
                     output_dir=None):
    """
    Runs the full PSMA PET segmentation pipeline.

    Args:
        input_dir (str): Directory containing PSMA PET and CT files.
        output_dir (str): Directory to save the segmentation results.
        file_format (str): Input file format, either "dicom" or "nifti". Defaults to "dicom".
        suv_threshold (float): SUV threshold for segmentation refinement. Defaults to 3.

    """
    
    weights_dir = setup_psmasegmentator()

    download_fold_weights_via_api(output_dir = weights_dir)

    nnUNet_predict_image(weights_dir, pet_dir, ct_dir, output_dir,
                         device='cuda', step_size=0.5, use_tta=True, verbose=False)
    