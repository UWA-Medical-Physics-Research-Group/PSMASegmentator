"""
@author: Jake Kendrick

"""
import pydicom
from datetime import datetime
import numpy as np
import os
import math
import pathlib
import SimpleITK as sitk


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


def read_dicom_image(dicom_path):
    """Read a DICOM image series
    Args:
        dicom_path (str|pathlib.Path): Path to the DICOM series to read
    Returns:
        sitk.Image: The image as a SimpleITK Image
    """
    dicom_images = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(str(dicom_path))
    return sitk.ReadImage(dicom_images)


def convertPET2SUV (dicom_dir, save_dir):
    
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
    
    #Write the image
    sitk.WriteImage(pet_image, save_dir + '.nii.gz')
    
    return pet_image