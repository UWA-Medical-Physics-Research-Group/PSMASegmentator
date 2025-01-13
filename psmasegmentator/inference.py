import os
import tempfile
import shutil
from pathlib import Path
import nibabel as nib
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.file_path_utilities import get_output_folder
from dicom import convertPET2SUV, read_dicom_image

def initialise_predictor(model_folder, device='cuda', step_size=0.5, use_tta=False, verbose=False):
    """
    Initialises the nnUNetPredictor for inference.

    Args:
        model_folder (str): Path to the trained model folder.
        device (str): Device for inference ('cuda', 'cpu', or 'mps'). Defaults to 'cuda'.
        step_size (float): Step size for sliding window prediction. Defaults to 0.5.
        use_tta (bool): Whether to use test-time augmentation (mirroring). Defaults to False.
        verbose (bool): Whether to enable verbose output. Defaults to False.

    Returns:
        nnUNetPredictor: Configured nnUNetPredictor object.
    """
    assert device in ['cuda', 'cpu', 'mps'] or isinstance(device, torch.device), (
        f"Invalid device specified: {device}. Must be 'cuda', 'cpu', 'mps', or a valid torch.device."
    )

    if device == 'cpu':
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif device == 'cuda':
        torch.set_num_threads(1)
        device = torch.device('cuda')
    elif isinstance(device, torch.device):
        torch.set_num_threads(1)
        device = device
    else:
        device = torch.device('mps')

    predictor = nnUNetPredictor(
        tile_step_size=step_size,
        use_gaussian=True,
        use_mirroring=use_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=verbose,
        allow_tqdm=True
    )
    predictor.initialise_from_trained_model_folder(model_folder, 
                                                   use_folds=None,
                                                   checkpoint_name = "checkpoint_final.pth")
    return predictor

def is_dicom(input_path):
    """
    Determines if the input is a DICOM file or directory.

    Args:
        input_path (str or Path): Path to the input file or directory.

    Returns:
        bool: True if the input is in DICOM format, False otherwise.
    """
    input_path = Path(input_path)

    if input_path.is_dir():
        # Check if any file in the directory is a valid DICOM file
        for file in input_path.iterdir():
            try:
                pydicom.dcmread(file, stop_before_pixels=True)
                return True
            except Exception:
                continue
    else:
        try:
            pydicom.dcmread(input_path, stop_before_pixels=True)
            return True
        except Exception:
            pass

    return False

def predict_image(pet_input, ct_input, predictor, output_path):
    """
    Runs the prediction pipeline for PET and CT images.

    Args:
        pet_input (str): Path to the PET image (DICOM or NIfTI).
        ct_input (str): Path to the CT image (DICOM or NIfTI).
        predictor (nnUNetPredictor): Initialized nnUNetPredictor object.
        output_path (str): Path to save the predicted segmentation.

    Returns:
        None
    """
    pet_input = Path(pet_input)
    ct_input = Path(ct_input)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        # Process PET image
        if is_dicom(pet_input):
            pet_image = convertPET2SUV(pet_input, write_output=False)
        else:
            pet_image = sitk.ReadImage(str(pet_input))

        # Save PET image to temporary directory
        pet_temp_path = tmp_dir / "images2predict/PSMA01_0000.nii.gz"
        sitk.WriteImage(pet_image, str(pet_temp_path))

        # Process CT image
        if is_dicom(ct_input):
            print("Reading CT DICOM image...")
            ct_image = read_dicom_image(ct_input)
        else:
            print("Reading CT NIfTI image...")
            ct_image = sitk.ReadImage(str(ct_input))

        # Check if CT and PET images have matching spacing and size
        pet_spacing = pet_image.GetSpacing()
        pet_size = pet_image.GetSize()
        ct_spacing = ct_image.GetSpacing()
        ct_size = ct_image.GetSize()

        if pet_spacing == ct_spacing and pet_size == ct_size:
            print("CT and PET image spacing and sizes match. Skipping resampling.")
            resampled_ct_image = ct_image
        else:
            print("Resampling CT image to match PET image spacing and size...")
            resampled_ct_image = sitk.Resample(
                ct_image,
                pet_image,
                interpolator=sitk.sitkBSpline
            )

        # Save resampled CT image to temporary directory
        ct_temp_path = tmp_dir / "images2predict/PSMA01_0001.nii.gz"
        sitk.WriteImage(resampled_ct_image, str(ct_temp_path))

        print(f"Checking the directory of images: {tmp_dir}")

        print("Running prediction...")

        #Initialise the predictor
        predictor = initialise_predictor(predictor, 
                                         device='cuda', 
                                         step_size=0.5, 
                                         use_tta=False, 
                                         verbose=False)
        
        # Run prediction
        predictor.predict_from_files(list_of_lists_or_source_folder=tmp_dir / "images2predict",
                                     output_folder_or_list_of_truncated_output_files=output_path,
                                     save_probabilities=False,
                                     overwrite=True,
                                     num_processes_preprocessing=2,
                                     num_processes_segmentation_export=4)



