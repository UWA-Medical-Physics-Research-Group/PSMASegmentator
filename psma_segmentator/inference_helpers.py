"""
@author: Jake Kendrick (adapted by Joel Noble using code from Saffi Hunt's CTSegmentator)
"""
import pydicom
from datetime import datetime
import numpy as np
import os
import math
import pathlib
from pathlib import Path
import SimpleITK as sitk


def is_dicom(input_path):
    """
    Determines if the input is a DICOM file or directory.

    Args:
        input_path (str or Path): Path to the input file or directory.

    Returns:
        bool: True if the input is in DICOM format, False otherwise.
    """
    input_path = Path(input_path)
    if input_path.is_file():
        return _is_valid_dicom_file(input_path)
    
    # Check if any file in the directory is a valid DICOM
    return any(_is_valid_dicom_file(file) for file in input_path.iterdir())

def _is_valid_dicom_file(file_path):
    """Helper to determine if a single file is a valid DICOM."""
    try:
        pydicom.dcmread(file_path, stop_before_pixels=True)
        return True
    except Exception:
        return False


def read_dicom_image(dicom_path):
    """
    Reads a DICOM image series into a SimpleITK image.

    Args:
        dicom_path (str|pathlib.Path): Path to the DICOM series.

    Returns:
        sitk.Image: The image as a SimpleITK Image.
    """
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(str(dicom_path))
    reader.SetFileNames(dicom_series)
    return reader.Execute()


def dicom_to_nifti(dicom_series_path, nifti_output_dir):
    """
    Converts a DICOM series to NIfTI format and saves it.

    Args:
        dicom_series_path (str): Path to the DICOM series directory.
        nifti_output_dir (str): Directory where the NIfTI file will be saved.

    Returns:
        str: Path to the saved NIfTI file.
    """
    os.makedirs(nifti_output_dir, exist_ok=True)
    
    volume = read_dicom_image(dicom_series_path)
    volume = sitk.DICOMOrient(volume, "LPS")

    patient_id = get_patient_id(dicom_series_path)
    nifti_filename = f"{patient_id}.nii.gz"
    output_nifti_path = os.path.join(nifti_output_dir, nifti_filename)

    sitk.WriteImage(volume, output_nifti_path)
    return output_nifti_path

def get_patient_id(input):
    """
    Retrieve the pateint ID from the first dicom in the folder of inputs
    Returns: 
        patient_id (str)
    """
    for file in os.listdir(input):
        if file.lower().endswith(('.dcm',)):  # Often no extension or .dcm
            file_path = os.path.join(input, file)
            try:
                dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
                return dcm.PatientID
            except Exception:
                continue
    raise ValueError(f"No valid DICOM files found in {input}")


def get_suv_bw_scale_factor(ds):
    
    '''
    Calculates the SUV body weight scale factor for individual PET axial slice.
    
    Inputs:
        ds (pydicom.dataset): PyDicom dataset
        
    Outputs: 
        suv_bw_scale_factor: SUV scale factor to be applied to the images.
    '''
    
    assert ds.Modality == "PT"
    assert "DECY" in ds.CorrectedImage
    assert "ATTN" in ds.CorrectedImage
    assert "START" in ds.DecayCorrection
    assert ds.Units == "BQML"
    assert "SeriesTime" in ds
    
    half_life = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
    
    series_date_time = ds.SeriesDate + "_" + ds.SeriesTime
        
    if "." in series_date_time:
        series_date_time = series_date_time[:-(len(series_date_time) - series_date_time.index("."))]
        
    series_date_time = datetime.strptime(series_date_time, "%Y%m%d_%H%M%S")
        
    start_time = (ds.SeriesDate + "_" + ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime)    
        
    if "." in start_time:
        start_time = start_time[: -(len(start_time) - start_time.index("."))]
            
    start_time = datetime.strptime(start_time, "%Y%m%d_%H%M%S")
    
    decay_time = (series_date_time - start_time).seconds
    injected_dose = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
    decayed_dose = injected_dose * pow(2, -decay_time / half_life)
    patient_weight = float(ds.PatientWeight)
    suv_bw_scale_factor = patient_weight * 1000 / decayed_dose
    
    return suv_bw_scale_factor


def convertPET2SUV (dicom_dir, save_dir, write_output = False):
    
    '''
    Converts PET image voxels into SUV units, and saves as Nifty format (.nii.gz) in the
    specified output directory.
    
    Inputs: 
        dicom_dir (path): Directory to the DICOM folder.
        save_dir (path): Directory to save resulting Nifty file.
        
    Outputs:
        pet_image (SimpleITK Image Object): SUV converted PET image is returned.
    '''
    
    #Get the dicom files in the directory (this will skip files like metacache.mim, for example)
    root_path = pathlib.Path(dicom_dir)
    dicom_file_list = [p for p in root_path.glob("**/*")
                    if p.name.lower().endswith(".dcm") or p.name.lower().endswith(".dc3")]
    
    #Read in the first file using Pydicom to get our metadata
    ds = pydicom.read_file(dicom_file_list[0], force=True)
    
    pet_image = read_dicom_image(dicom_dir) #Automatically applies rescape slope and intercept.
    suv_factor = get_suv_bw_scale_factor(ds)
    
    #Convert our PET image
    pet_image *= suv_factor
    
    if write_output == True:    
        sitk.WriteImage(pet_image, save_dir + '.nii.gz')
    
    return pet_image
