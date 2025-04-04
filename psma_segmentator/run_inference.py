import os
import tempfile
from pathlib import Path
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import torch
import pydicom
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from psma_segmentator.inference_helpers import convertPET2SUV, read_dicom_image, is_dicom, dicom_to_nifti, get_patient_id

def prepare_pet_ct_images(ct_input: str, 
                            pet_input: str, 
                            tmp_dir: str,
                            # case_name: str
                        ):
    """
    Prepares PET and CT images for nnUNet inference by:
    - Converting DICOM to NIfTI if necessary
    - Converting PET to SUV if applicable
    - Resampling CT to match PET
    - Saving final preprocessed images in a temporary directory
    
    Args:
        ct_input (str): Path to the input CT image or DICOM directory
        pet_input (str): Path to the input PET image or DICOM directory
        tmp_dir (str): Path to the temporary directory where preprocessed images will be saved
    
    Returns:
        (tuple): Paths to the preprocessed PET and CT images
    """
    tmp_dir = Path(tmp_dir)
    
    # Process PET image
    if is_dicom(pet_input):
        print("Converting PET DICOM to NIfTI and converting to SUV...")
        pet_image = convertPET2SUV(dicom_dir=pet_input, 
                                    save_dir=tmp_dir
                                    )
    else:
        print("Reading PET NIfTI image...")
        pet_image = sitk.ReadImage(str(pet_input))
    
    pet_temp_path = tmp_dir / "PSMA_0001.nii.gz"
    sitk.WriteImage(pet_image, str(pet_temp_path))
    
    # Process CT image
    if is_dicom(ct_input):
        print("Reading CT DICOM image...")
        ct_image = read_dicom_image(ct_input)
    else:
        print("Reading CT NIfTI image...")
        ct_image = sitk.ReadImage(str(ct_input))
    
    # Ensure CT and PET match in spacing and size
    if pet_image.GetSpacing() == ct_image.GetSpacing() and pet_image.GetSize() == ct_image.GetSize():
        print("CT and PET image spacing and sizes match. Skipping resampling.")
        resampled_ct_image = ct_image
    else:
        print("Resampling CT image to match PET image spacing and size...")
        resampled_ct_image = sitk.Resample(
            ct_image,
            pet_image,
            interpolator=sitk.sitkBSpline
        )
    
    ct_temp_path = tmp_dir / "PSMA_0000.nii.gz"
    sitk.WriteImage(resampled_ct_image, str(ct_temp_path))
    
    return str(ct_temp_path), str(pet_temp_path)


def nnUNet_predict_image(model_folder, 
                            ct_input, pet_input, 
                            output_path, 
                            device, step_size=0.5, 
                            use_tta=False, verbose=False):
    """
    Runs inference using nnUNet for a single PET and CT image pair.
    """
    assert device in ['cuda', 'cpu', 'mps'] or isinstance(device, torch.device), "Invalid device specified."
    
    if device == 'cpu':
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif device == 'cuda':
        torch.set_num_threads(1)
        device = torch.device('cuda')
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
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=None, checkpoint_name="checkpoint_final.pth")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        ct_temp_path, pet_temp_path = prepare_pet_ct_images(ct_input, pet_input, tmp_dir)

        print(f"Checking the directory of images: {tmp_dir}")
        print(f"CT image: {ct_temp_path}")
        print(f"PET image: {pet_temp_path}")
        print(f"Output path: {output_path}")
        
        print("Running prediction...")
        predictor.predict_from_files(
            list_of_lists_or_source_folder=str(tmp_dir),
            output_folder_or_list_of_truncated_output_files=str(output_path),
            save_probabilities=False,
            overwrite=True
        )