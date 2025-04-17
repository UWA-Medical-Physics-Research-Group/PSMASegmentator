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
from platipy.dicom.io.rtstruct_to_nifti import convert_rtstruct
from tqdm import tqdm


def shorten_path(path, max_parts=2):
    path = Path(path)
    parts = path.parts
    if len(parts) <= max_parts * 2 + 1:
        return str(path)

    start = Path(*parts[:max_parts])
    end = Path(*parts[-max_parts:])
    return str(start / "..." / end)


def get_modality_dirs_and_validate_pet(study_dir):
    """
    Checks all DICOM dirs in a study, groups them by modality (CT/PT), and validates PET if found.
    Returns a dict like {'CT': [Path], 'PT': [Path]} or empty dict if PET is invalid.
    """
    dicom_series = defaultdict(list)

    for dicom_dir in study_dir.iterdir():
        if not dicom_dir.is_dir():
            continue

        modalities_present = set()
        representative_ds = None

        for file in dicom_dir.glob("*.dcm"):
            try:
                ds = pydicom.dcmread(file, stop_before_pixels=True)
                modality = ds.Modality.upper()
                modalities_present.add(modality)
                if representative_ds is None:
                    representative_ds = ds  # Just one is enough
            except Exception:
                continue

        if len(modalities_present) > 1:
            print(f"Multiple modalities found in {shorten_path(dicom_dir)}. Skipping entire study.")
            return {}  # Abort processing for this study

        if modality == "PT":
            try:
                assert "DECY" in ds.CorrectedImage
                assert "ATTN" in ds.CorrectedImage
                assert ds.DecayCorrection == "START"
                assert ds.Units == "BQML"
                assert hasattr(ds, "SeriesTime")
                assert hasattr(ds, "SeriesDate")
                assert hasattr(ds, "PatientWeight")
                assert hasattr(ds, "RadiopharmaceuticalInformationSequence")
                rds = ds.RadiopharmaceuticalInformationSequence[0]
                assert hasattr(rds, "RadionuclideHalfLife")
                assert hasattr(rds, "RadionuclideTotalDose")
                assert hasattr(rds, "RadiopharmaceuticalStartTime")
            except Exception:
                print(f"Invalid PET series found in {shorten_path(study_dir)}. Skipping entire study.")
                return {}  # Abort processing for this study

        # Fast header-only check with SimpleITK (no image load)
        try:
            series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir))
            # if not series_IDs:
            #     raise ValueError("No series IDs found")

            series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), series_IDs[0])
            # if not series_file_names:
            #     raise ValueError("No files in series")
            
            # Just test the first slice
            test_reader = sitk.ImageFileReader()
            test_reader.SetFileName(series_file_names[0])
            test_reader.ReadImageInformation()
        except Exception as e:
            print(f"SimpleITK header read failed in {shorten_path(dicom_dir)}. Skipping entire study. Error: {str(e)}")
            return {}

        dicom_series[modality] = (dicom_dir, ds)
    # if 'CT' and 'PT' not in dicom_series.keys():
    if not {'CT', 'PT'}.issubset(dicom_series.keys()):
        print(f"CT and/or PET series not found in {shorten_path(study_dir)}. Skipping entire study.")
        return {} # Abort processing for this study
    
    return dicom_series


def dicom_to_nifti(dicom_dir: str, nifti_path: str, 
                    is_pet: bool, ds):
    """
    Converts a DICOM series to NIfTI format and saves it.

    Args:
        dicom_dir (str): Path to the DICOM series directory.
        nifti_output_dir (str): Directory where the NIfTI file will be saved.

    Returns:
        str: Path to the saved NIfTI file.
    """    
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    reader.SetFileNames(dicom_series)
    volume = reader.Execute()
    volume = sitk.DICOMOrient(volume, "LPS")
    size = volume.GetSize() # Get wise

    if is_pet:
        suv_factor = get_suv_bw_scale_factor(ds)
        # Convert the PET image to SUV
        volume *= suv_factor

    sitk.WriteImage(volume, nifti_path)
    return size


def get_suv_bw_scale_factor(ds):
    
    '''
    Calculates the SUV body weight scale factor for individual PET axial slice.
    
    Inputs:
        ds (pydicom.dataset): PyDicom dataset
        
    Outputs: 
        suv_bw_scale_factor: SUV scale factor to be applied to the images.
    '''
    radiopharm_info = ds.RadiopharmaceuticalInformationSequence[0]
    half_life = float(radiopharm_info.RadionuclideHalfLife)
    
    series_date_time = ds.SeriesDate + "_" + ds.SeriesTime
        
    if "." in series_date_time:
        series_date_time = series_date_time[:-(len(series_date_time) - series_date_time.index("."))]
        
    series_date_time = datetime.strptime(series_date_time, "%Y%m%d_%H%M%S")
        
    start_time = (ds.SeriesDate + "_" + radiopharm_info.RadiopharmaceuticalStartTime)    
        
    if "." in start_time:
        start_time = start_time[: -(len(start_time) - start_time.index("."))]
            
    start_time = datetime.strptime(start_time, "%Y%m%d_%H%M%S")
    
    decay_time = (series_date_time - start_time).seconds
    injected_dose = float(radiopharm_info.RadionuclideTotalDose)
    decayed_dose = injected_dose * pow(2, -decay_time / half_life)
    patient_weight = float(ds.PatientWeight)
    suv_bw_scale_factor = patient_weight * 1000 / decayed_dose
    
    return suv_bw_scale_factor


def pre_process(input_dir, incl_rtstructs, verbose, overwrite, output_seg_dir):
    input_dir = Path(input_dir)
    case_dict = defaultdict(list)

    nii_files = list(input_dir.rglob("*.nii.gz"))
    if nii_files:
        # Already in NIfTI format
        print(f"Existing NIfTI files found in {input_dir}. Using these files directly for inference.")
        for f in nii_files:
            base = f.stem.rsplit('_000', 1)[0]
            seg_path = f"{output_seg_dir}/{base}.nii.gz"
            if Path(seg_path).exists() and not overwrite:
                print(f"Skipping {base}: segmentation already exists at {shorten_path(seg_path)}.")
                continue
            case_dict[base].append(str(f))
        list_of_lists = [sorted(files) for files in case_dict.values()]
        if not list_of_lists:
            print(f"All segmentations already exist in {output_seg_dir}. Nothing to process.")
        return list_of_lists
    
    elif any(f.suffix == '.dcm' for f in input_dir.rglob('*')):
        # Otherwise process DICOM
        print(f"DICOM files detected in {input_dir}. Converting to NIfTI format.")
        output_dir = input_dir.parent / f"{input_dir.name}_preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)

        if incl_rtstructs:
            output_dir_structs = input_dir.parent / f"{input_dir.name}_gt_segmentations"
            output_dir_structs.mkdir(parents=True, exist_ok=True)

        case_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        for case_dir in tqdm(case_dirs, desc="Pre-processing cases"):
            # Set case_name to the directory name
            case_name = case_dir.name
            if verbose:
                print(f"\nCase: {case_name}")

            seg_path = f"{output_seg_dir}/{case_name}.nii.gz"
            if Path(seg_path).exists() and not overwrite:
                print(f"Skipping {case_name}: segmentation already exists.")
                continue

            # Early skip if both CT and PET outputs exist
            ct_nifti_path = output_dir / f"{case_name}_0000.nii.gz"
            pt_nifti_path = output_dir / f"{case_name}_0001.nii.gz"
            output_dir_struct = output_dir_structs / case_name if incl_rtstructs else None

            if ct_nifti_path.exists() and pt_nifti_path.exists() and not overwrite:
                if incl_rtstructs:
                    if output_dir_struct.exists() and any(output_dir_struct.glob("*.nii.gz")):
                        print(f"Skipping {case_name} (preprocessed CT, PET, and RTSTRUCTs found).")
                        case_dict[case_name].extend([str(ct_nifti_path), str(pt_nifti_path)])
                        continue
                    else:
                        print(f"Skipping {case_name} (preprocessed CT and PET found) but no pre-processed RTSTRUCTs found.")
                        case_dict[case_name].extend([str(ct_nifti_path), str(pt_nifti_path)])
                        continue
                else:
                    print(f"Skipping {case_name} (preprocessed CT and PET found).")
                    case_dict[case_name].extend([str(ct_nifti_path), str(pt_nifti_path)])
                    continue
            elif ct_nifti_path.exists() and pt_nifti_path.exists() and overwrite:
                print(f"Overwriting preprocessed files for {case_name}.")

            for study_dir in case_dir.iterdir():
                sizes = {}
                if not study_dir.is_dir():
                    continue

                dicom_series = get_modality_dirs_and_validate_pet(study_dir)
                if not dicom_series: # the study failed the check
                    continue

                for modality in dicom_series.keys():
                    dicom_dir, ds = dicom_series[modality]
                    is_pet = False
                    
                    if modality in ['CT', 'PT']:
                        if modality == 'CT':
                            nifti_path = ct_nifti_path
                        else:
                            nifti_path = pt_nifti_path
                            is_pet = True
                        print(f"Found {modality} DICOM series in {shorten_path(dicom_dir)}, converting to NIfTI.")

                        vol_size = dicom_to_nifti(dicom_dir, nifti_path, 
                                                    is_pet, ds)
                        case_dict[case_name].append(str(nifti_path))
                        sizes[modality] = vol_size
                    
                    elif modality == 'RTSTRUCT':
                        dcm_rt_file = next(dicom_dir.glob("*.dcm"), None)
                        if incl_rtstructs:
                            # Convert RTSTRUCT to NIfTI
                            output_dir_struct = output_dir_structs / case_name
                            os.makedirs(output_dir_struct, exist_ok=True)
                            if dcm_rt_file is None:
                                print(f"No RTSTRUCT DICOM file found in {shorten_path(dcm_rt_file)}, skipping ground truth pre-processing for this case.")
                                continue
                            convert_rtstruct(dcm_img=dicom_dir, 
                                                dcm_rt_file=dcm_rt_file, 
                                                # prefix=case_name,
                                                output_dir=output_dir_struct)
                            print(f"{modality} converted to NIfTI masks saved at {shorten_path(output_dir_struct)}")
                        else:
                            print(f"RTSTRUCT found in {shorten_path(dcm_rt_file)}, but RTSTRUCT pre-processing is disabled. Skipping.")
                    else:
                        print(f"Unsupported modality: {modality} in {shorten_path(study_dir)}. Skipping.") # how'd this modality get in here?!?

                if sizes['CT'] != sizes['PT']:
                    if verbose:
                        print(f"CT and PET sizes ({sizes['CT']} | {sizes['PT']}) do not match for {case_name}. Resampling required.")
                    # Load output nifti files for case and check if resampling required
                    ct_path = next(p for p in case_dict[case_name] if "_0000" in p)
                    pet_path = next(p for p in case_dict[case_name] if "_0001" in p)

                    ct_img = sitk.ReadImage(ct_path)
                    pet_img = sitk.ReadImage(pet_path)

                    resampled_ct = sitk.Resample(ct_img, pet_img)
                        
                    sitk.WriteImage(resampled_ct, str(ct_path))
                    if verbose:
                        print(f"Saved NIfTI CT path: {shorten_path(ct_path)}")
                        print(f"Saved NIfTI PET path: {shorten_path(pet_path)}")

        list_of_lists = [sorted(files) for files in case_dict.values()]
        if not list_of_lists:
            print(f"All segmentations already exist in {output_seg_dir}. Nothing to process.")
        return list_of_lists
    
    else:
        raise ValueError(f"No valid NIfTI or DICOM files found in {input_dir}.")    
