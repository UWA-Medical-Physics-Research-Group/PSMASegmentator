import sys
import os
from pathlib import Path
import SimpleITK as sitk
import numpy as np


def psmasegmentator (input_dir=None, output_dir=None, file_format="dicom", suv_threshold=3):
    """
    Runs the full PSMA PET segmentation pipeline.

    Args:
        input_dir (str): Directory containing PSMA PET and CT files.
        output_dir (str): Directory to save the segmentation results.
        file_format (str): Input file format, either "dicom" or "nifti". Defaults to "dicom".
        suv_threshold (float): SUV threshold for segmentation refinement. Defaults to 3.

    Returns:
        dict: Paths to the output segmentation files.

    """
    print("Pipeline function is running...")
    