import os
import shutil
import tempfile
from pathlib import Path
import pydicom
from datetime import datetime
import numpy as np
import math
import SimpleITK as sitk
from collections import defaultdict

def dicom_and_modality_check(input_path):
    """
    Returns modality + pydicom dataset for first valid DICOM found.

    Raises ValueError only if no readable DICOM files found.
    """
    input_path = Path(input_path)

    if not input_path.is_dir():
        raise ValueError(f"Expected directory but got {input_path}")

    for f in sorted(input_path.iterdir()):
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            return ds.Modality.upper(), ds
        except Exception:
            continue  # Not a dicom / unreadable

    raise ValueError(f"No valid DICOM files found in {input_path}")

def dicom_to_nifti(dicom_dir: str, output_dir: str, 
                    case_name: str, modality, ds):
    """
    Converts a DICOM series to NIfTI format and saves it.

    Args:
        dicom_dir (str): Path to the DICOM series directory.
        nifti_output_dir (str): Directory where the NIfTI file will be saved.

    Returns:
        str: Path to the saved NIfTI file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    reader.SetFileNames(dicom_series)
    volume = reader.Execute()
    volume = sitk.DICOMOrient(volume, "LPS")
    size = volume.GetSize() # Get wise

    if modality == 'CT':
        print(f"Processing CT DICOM series in {dicom_dir}.")
        nifti_filename = f"{case_name}_0000.nii.gz"

    elif modality == 'PT':
        print(f"Processing PET DICOM series in {dicom_dir}.")
        suv_factor = get_suv_bw_scale_factor(ds)
        # Convert the PET image to SUV
        volume *= suv_factor
        nifti_filename = f"{case_name}_0001.nii.gz"

    output_nifti_path = os.path.join(output_dir, nifti_filename)
    sitk.WriteImage(volume, output_nifti_path)
    return output_nifti_path, size


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


def pre_process(input_dir):
    input_dir = Path(input_dir)
    case_dict = defaultdict(list)

    nii_files = list(input_dir.rglob("*.nii.gz"))
    if nii_files:
        # Already in NIfTI format
        print(f"Existing NIfTI files found in {input_dir}. Using these files directly for inference.")
        for f in nii_files:
            base = f.stem.rsplit('_000', 1)[0]
            case_dict[base].append(str(f))
        list_of_lists = [sorted(files) for files in case_dict.values()]
        return list_of_lists
    
    elif any(f.suffix == '.dcm' for f in input_dir.rglob('*')):
        # Otherwise process DICOM
        print(f"DICOM files detected in {input_dir}. Converting to NIfTI format.")
        output_dir = input_dir.parent / f"{input_dir.name}_nifti"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect all cases
        for case_dir in input_dir.iterdir():
            # Set case_name to the directory name
            case_name = case_dir.name
            sizes = {}
            print(f"\nProcessing case: {case_name}")
            if case_dir.is_dir():
                for study in case_dir.iterdir():
                    if study.is_dir():
                        for dicom_dir in study.iterdir():
                            modality, ds = dicom_and_modality_check(dicom_dir)
                            if modality not in ['CT', 'PT']:
                                print(f"Ignoring unsupported modality: {modality} in {dicom_dir}")
                                continue
                            else:
                                print(f"Found {modality} DICOM series in {dicom_dir}, converting to NIfTI.")
                                # Convert DICOM to NIfTI
                                output_nifti_path, size = dicom_to_nifti(dicom_dir, output_dir, 
                                                        case_name, modality, ds)
                                case_dict[case_name].append(str(output_nifti_path))
                                sizes[modality] = size
            if sizes['CT'] != sizes['PT']:
                print(f"CT and PET sizes ({sizes['CT']} | {sizes['PT']}) do not match for {case_name}. Resampling required.")
                # Load output nifti files for case and check if resampling required
                ct_path = next(p for p in case_dict[case_name] if "_0000" in p)
                pet_path = next(p for p in case_dict[case_name] if "_0001" in p)
                print(f"NIfTI CT path: {ct_path}, NIfTI PET path: {pet_path}")

                ct_img = sitk.ReadImage(ct_path)
                pet_img = sitk.ReadImage(pet_path)

                resampled_ct = sitk.Resample(ct_img, pet_img)
                    
                sitk.WriteImage(resampled_ct, str(ct_path))

        list_of_lists = [sorted(files) for files in case_dict.values()]
        return list_of_lists
    
    else:
        raise ValueError(f"No valid NIfTI or DICOM files found in {input_dir}.")    
