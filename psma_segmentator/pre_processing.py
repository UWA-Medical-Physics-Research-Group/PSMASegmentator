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
import subprocess
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

        tomographic_modalities = {"CT", "PT", "MR"}
        tomo_modalities_found = modalities_present & tomographic_modalities
        non_tomo_modalities = modalities_present - tomographic_modalities

        if len(tomo_modalities_found) > 1:
            print(f"ERROR: Conflicting tomographic modalities in {shorten_path(dicom_dir)}: {tomo_modalities_found}. Skipping entire study.")
            return None

        if tomo_modalities_found and non_tomo_modalities:
            print(f"WARNING: Tomographic modality {tomo_modalities_found} mixed with non-tomographic {non_tomo_modalities} in {shorten_path(dicom_dir)}. Proceeding.")

        if modality == "PT":
            missing_tags = []

            # Required tags and conditions
            try:
                corrected_image = representative_ds.CorrectedImage
                if "DECY" not in corrected_image:
                    missing_tags.append("CorrectedImage: 'DECY' missing")
                if "ATTN" not in corrected_image:
                    missing_tags.append("CorrectedImage: 'ATTN' missing")
            except Exception:
                missing_tags.append("CorrectedImage")

            if getattr(representative_ds, "DecayCorrection", None) != "START":
                missing_tags.append("DecayCorrection != 'START'")

            if getattr(representative_ds, "Units", None) != "BQML":
                missing_tags.append("Units != 'BQML'")

            for attr in ["SeriesTime", "SeriesDate", "PatientWeight", "RadiopharmaceuticalInformationSequence"]:
                if not hasattr(representative_ds, attr):
                    missing_tags.append(attr)

            rds = getattr(representative_ds, "RadiopharmaceuticalInformationSequence", [{}])[0]
            for attr in ["RadionuclideHalfLife", "RadionuclideTotalDose", "RadiopharmaceuticalStartTime"]:
                if not hasattr(rds, attr):
                    missing_tags.append(f"RadiopharmaceuticalInformationSequence[0].{attr}")

            if missing_tags:
                print(f"Invalid PET series found in {shorten_path(dicom_dir)}. Missing or incorrect: {missing_tags}. Skipping entire study.")
                return None

        # Only attempt SimpleITK read for CT and PT
        if modality in {"CT", "PT"}:
            try:
                series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir))
                series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), series_IDs[0])
                test_reader = sitk.ImageFileReader()
                test_reader.SetFileName(series_file_names[0])
                test_reader.ReadImageInformation()
            except Exception as e:
                print(f"SimpleITK header read failed in {shorten_path(dicom_dir)}. Skipping entire study. Error: {str(e)}")
                return None

        dicom_series[modality] = (dicom_dir, ds)

    if not {'CT', 'PT'}.issubset(dicom_series.keys()):
        print(f"CT and/or PET series not found in {shorten_path(study_dir)}. Skipping entire study.")
        return None

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


def plastimatch_rtstruct_to_nifti(ct_dcm, rt_dcm, output_dir_struct,
                                    rename_map=None):
    """
    Convert RTSTRUCT DICOM file into separate NIfTI masks using Plastimatch.

    Args:
        ct_dcm (Path or str): Path to a single DICOM CT image file (used as reference).
        rt_dcm (Path or str): Path to the RTSTRUCT DICOM file.
        output_dir_struct (Path or str): Path to output directory for the structure NIfTI files.
        rename_map (dict): Optional dictionary mapping original structure names (or keywords) to new filenames.

    Raises:
        RuntimeError: If conversion fails at any step.
    """
    ct_dcm = Path(ct_dcm)
    rt_dcm = Path(rt_dcm)

    try:
        command = [
            'plastimatch', 'convert',
            '--input', str(rt_dcm),
            '--referenced-ct', str(ct_dcm),
            '--output-prefix', str(output_dir_struct) + os.sep,
            '--prefix-format', 'nii.gz',
            '--prune-empty',
        ]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Plastimatch RTSTRUCT→NIfTI conversion failed: {e}")

    for file in output_dir_struct.glob("*.nii.gz"):
        original_filename = file.name
        base, ext = original_filename.split(".", 1)  # Split at first dot only

        # Apply rename_map if any match found
        new_base = None
        if rename_map:
            for key, val in rename_map.items():
                if key.lower() in base.lower():
                    new_base = val
                    break

        if not new_base:
            new_base = base.lower().replace("-", "").replace(" ", "_").replace("`", "")

        new_name = new_base + "." + ext  # Preserve original extension exactly
        new_path = file.parent / new_name

        if original_filename != new_name:
            file.rename(new_path)

def process_dicom(dicom_series, modality, nifti_path, sizes):
    """
    Process a DICOM series and convert it to NIfTI format.

    Args:
        dicom_series (dict): Dictionary containing DICOM series paths.
        modality (str): Modality type ('CT' or 'PT').
        nifti_path (str): Path to save the NIfTI file.
        sizes (dict): Dictionary to store the size of the converted image.

    Returns:
        None
    """
    dicom_dir = dicom_series[modality][0]
    ds = dicom_series[modality][1]
    is_pet = modality == "PT"
    
    size = dicom_to_nifti(dicom_dir, nifti_path, is_pet, ds)
    sizes[modality] = size


def pre_process(input_path, incl_rtstructs, 
                output_pred_dir, 
                verbose, overwrite):

    if any(f.suffix == ".dcm" for f in input_path.rglob("*")):
        print(f"Input path {shorten_path(input_path)} contains DICOM files.")
        handling_dicom = True
        output_prepro_dir = str(input_path.parent / f"{input_path.name}_preprocessed")
        os.makedirs(output_prepro_dir, exist_ok=True)
    else:
        print(f"Input path {shorten_path(input_path)} contains NIfTI files.")
        handling_dicom = False
        output_prepro_dir = str(input_path)

    case_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    if case_dirs:
        print(f"Found case directories in input path, assuming DICOM input.")
        
        output_dir_structs = (input_path.parent / f"{input_path.name}_structs") if incl_rtstructs else None
        output_dir_gts = (input_path.parent / f"{input_path.name}_gt_segmentations") if incl_rtstructs else None
        if incl_rtstructs:
            output_dir_structs.mkdir(parents=True, exist_ok=True)
            output_dir_gts.mkdir(parents=True, exist_ok=True)

        # Filter to only cases needing prediction
        if not overwrite:
            case_dirs_to_predict = find_predicted(case_dirs, output_pred_dir, 
                                                    mode='case_dirs', verbose=verbose)
        else:
            case_dirs_to_predict = case_dirs

        if not case_dirs_to_predict:
            print("All cases have existing predictions. Nothing to process.")
            return [], output_prepro_dir

        # Among cases needing prediction, check which are already preprocessed
        if not overwrite:
            not_preprocessed, already_preprocessed = find_preprocessed(case_dirs_to_predict, output_prepro_dir, incl_rtstructs, 
                                                                        output_dir_gts, verbose)
        else:
            not_preprocessed = case_dirs_to_predict # If overwriting, all are considered not preprocessed
            already_preprocessed = []

        # print(f"[DEBUG] Remaining case directories to process: {not_preprocessed}")

        # Handle DICOM input
        if handling_dicom:
            newly_preprocessed = _handle_dicom_data(not_preprocessed,
                                                    output_prepro_dir, 
                                                    incl_rtstructs, output_dir_structs, output_dir_gts, 
                                                    verbose, overwrite, delete_structs_dir=False)
            return already_preprocessed + (newly_preprocessed or []), output_prepro_dir
    else:
        # Handle NIfTI input
        nii_files = list(input_path.rglob("*.nii.gz"))
        if not overwrite: # only check for predicted NIfTI files if not overwriting
            nii_files = find_predicted(nii_files, output_pred_dir,
                                        mode='nii_files', verbose=verbose)
        if nii_files:
            newly_processed = _handle_existing_nifti_files(nii_files, output_prepro_dir, 
                                                            overwrite, verbose)
            return newly_processed, output_prepro_dir # + preprocessed
        else:
            print("All NIfTI output files already exist and/or overwrite = False. Nothing to predict.")
            return [], output_prepro_dir

    raise ValueError(f"No valid NIfTI or DICOM files found in {input_path}.")


# Sub-functions

def find_predicted(input_dir, output_pred_dir, mode, verbose=True):
    remaining = [] # List of files/directories that need to be processed

    if mode == "case_dirs": # input_dir is case_dirs
        for case_dir in input_dir:
            case_name = case_dir.name
            pred_file = Path(os.path.join(output_pred_dir, f"{case_name}.nii.gz"))
            if pred_file.exists():
                if verbose:
                    print(f"Skipping {case_name}: prediction exists at {shorten_path(pred_file)}.")
                continue
            remaining.append(case_dir)

    elif mode == "nii_files":
        for nii_file in input_dir: # input_dir is nii_files
            case_name = os.path.basename(nii_file).split("_000")[0]
            pred_file = Path(os.path.join(output_pred_dir, f"{case_name}.nii.gz"))
            if pred_file.exists():
                if verbose:
                    print(f"Skipping {case_name}: prediction exists at {shorten_path(pred_file)}.")
                continue
            else:
                if verbose:
                    print(f"Processing {case_name}: prediction does not exist at {shorten_path(pred_file)}.")
            remaining.append(nii_file)

    return remaining

def find_preprocessed(case_dirs, output_prepro_dir, incl_rtstructs, output_dir_gts, verbose):
    if not case_dirs:
        print(f"No case directories to check.")
        return []

    remaining_dirs = []
    preprocessed = []  # List to store already preprocessed cases

    for case_dir in case_dirs:
        case_name = case_dir.name
        ct_path = Path(output_prepro_dir) / f"{case_name}_0000.nii.gz"
        pt_path = Path(output_prepro_dir) / f"{case_name}_0001.nii.gz"

        ct_done, pt_done, gt_done = _already_preprocessed(
            ct_path, pt_path, output_dir_gts, 
            case_name, incl_rtstructs, verbose
        )

        if not (ct_done and pt_done and (not incl_rtstructs or gt_done)):
            if verbose:
                print(f"[DEBUG] Case {case_name} is NOT fully preprocessed. Will process.")
            remaining_dirs.append(case_dir)
        else:
            if verbose:
                print(f"[DEBUG] Case {case_name} is fully preprocessed. Skipping.")
            preprocessed.append(case_name)

    return remaining_dirs, preprocessed

def _handle_existing_nifti_files(nii_files, output_prepro_dir, 
                                    overwrite, verbose):
    case_dict = defaultdict(list)
    print(f"Existing NIfTI files found. Collating and using these directly for inference.")

    for f in nii_files:
        print(f"Processing {shorten_path(f)}")
        base = f.stem.rsplit('_000', 1)[0]
        nii_path = Path(output_prepro_dir) / f"{base}.nii.gz"

        # Check if ct and pet niftis exist and are the same size
        ct_path = Path(output_prepro_dir) / f"{base}_0000.nii.gz"
        pt_path = Path(output_prepro_dir) / f"{base}_0001.nii.gz"

        if ct_path.exists() and pt_path.exists():
            ct_img = sitk.ReadImage(str(ct_path))
            pt_img = sitk.ReadImage(str(pt_path))
            if ct_img.GetSize() != pt_img.GetSize():
                if verbose:
                    print(f"CT and PET sizes ({ct_img.GetSize()} | {pt_img.GetSize()}) do not match for {base}. Resampling required.")
                _resample_ct_to_pet(ct_path, pt_path, verbose=True)
        else:
            print(f"CT and/or PET NIfTI files not found for {base}. Skipping.")
            continue

        if nii_path.exists() and not overwrite:
            print(f"Skipping {base}: pre-processed NIfTI already exists at {shorten_path(nii_path)}.")
            continue
        case_dict[base].append(str(f))

    list_of_lists = [sorted(files) for files in case_dict.values()]
    if not list_of_lists:
        print(f"All pre-processed NIfTIs already exist in {output_prepro_dir}. Nothing to process.")
    return list_of_lists

def _handle_dicom_data(case_dirs,
                        # input_path, 
                        output_prepro_dir,
                        incl_rtstructs, output_dir_structs, output_dir_gts, 
                        verbose, overwrite, delete_structs_dir):
    # Make dir to store pre-processed NIfTIs in
    os.makedirs(output_prepro_dir, exist_ok=True)
    case_dict = defaultdict(list)

    # case_dirs = [d for d in input_path.iterdir() if d.is_dir()]

    for case_dir in tqdm(case_dirs, desc="Pre-processing cases"):
        case_name = case_dir.name
        if verbose:
            print("="*60)
            print(f"\n\nCase: {case_name}")
            print("="*60)

        for study_dir in case_dir.iterdir():
            if not study_dir.is_dir():
                continue

            dicom_series = get_modality_dirs_and_validate_pet(study_dir)
            if dicom_series is None:  # the study failed the check
                # Error message already printed in get_modality_dirs_and_validate_pet
                continue

            ct_path, pt_path = Path(f"{output_prepro_dir}/{case_name}_0000.nii.gz"), Path(f"{output_prepro_dir}/{case_name}_0001.nii.gz")
            # Regardless of further processing, add paths to case_dict for later inference

            ct_done, pt_done, gt_done = _already_preprocessed(ct_path, pt_path, output_dir_gts, 
                                                                    case_name, incl_rtstructs, verbose)

            # Skip if all required parts are done
            if (not overwrite) and (ct_done and pt_done and (not incl_rtstructs or gt_done)):
                print(f"Skipping {case_name} (preprocessed CT, PET{', GT' if incl_rtstructs else ''} found).") #at {shorten_path(output_prepro_dir)}).")
                # Add preprocessed paths to case_dict
                case_dict[case_name].extend([str(ct_path), str(pt_path)])
                continue

            sizes = {}

            if 'CT' in dicom_series and not ct_done:
                process_dicom(dicom_series, 'CT', ct_path, sizes)

            if 'PT' in dicom_series and not pt_done:
                process_dicom(dicom_series, 'PT', pt_path, sizes)

            if incl_rtstructs and not gt_done: # and 'RTSTRUCT' in dicom_series
                _process_rtstruct(dicom_series, study_dir, case_name, 
                                    output_dir_structs, output_dir_gts, verbose)

            if 'CT' in sizes and 'PT' in sizes and sizes['CT'] != sizes['PT']:
                if verbose:
                    print(f"CT and PET sizes ({sizes['CT']} | {sizes['PT']}) do not match for {case_name}. Resampling required.")
                _resample_ct_to_pet(ct_path, pt_path, verbose)

            # Add preprocessed paths to case_dict - CURRENTLY ASSUMES ONE STUDY PER CASE
            case_dict[case_name].extend([str(ct_path), str(pt_path)])

    # Optionally delete the intermediate output_dir_struct
    if delete_structs_dir:
        shutil.rmtree(output_dir_structs)
        if verbose:
            print(f"Deleted intermediate RTSTRUCT output directory {shorten_path(output_dir_structs)}")

    list_of_lists = [sorted(files) for files in case_dict.values()]
    if not list_of_lists:
        print(f"No valid PET/CT files collated from filtered case_dirs. Nothing to process.")
    return list_of_lists

def _already_preprocessed(ct_path, pt_path, output_dir_gts, 
                            case_name, incl_rtstructs, verbose=False):
    ct_done = ct_path.exists()
    pt_done = pt_path.exists()
    gt_done = False

    if incl_rtstructs:
        gt_file = output_dir_gts / f"{case_name}.nii.gz"
        gt_done = gt_file.exists() #and any(gt_dir.glob("*.nii.gz"))
        if verbose:
            print(f"Checking GT at {shorten_path(gt_file)}: exists={gt_file.exists()}, files_found={gt_done}")

    if verbose:
        print(f"Preprocessing check for {case_name}: CT={ct_done}, PT={pt_done}, GT={gt_done}")

    return ct_done, pt_done, gt_done

def _process_rtstruct(dicom_series, study_dir, case_name,
                        output_dir_structs, output_dir_gts,
                        verbose=False
                        ):
    ct_dir = dicom_series['CT'][0]
    ct_dcm = next(ct_dir.glob("*.dcm"), None)
    output_dir_struct = output_dir_structs / case_name
    os.makedirs(output_dir_struct, exist_ok=True)

    rtstruct_dirs = dicom_series.get('RTSTRUCT', [])
    rt_dcm = next(rtstruct_dirs[0].glob("*.dcm"), None) if rtstruct_dirs else None
    
    if not rt_dcm:
        print(f"No RTSTRUCT DICOM file found in {shorten_path(study_dir)}. Creating empty GT mask instead.")

        # Load full CT series for shape and metadata
        reader = sitk.ImageSeriesReader()
        series_file_names = reader.GetGDCMSeriesFileNames(str(ct_dir))
        reader.SetFileNames(series_file_names)
        ct_image = reader.Execute()

        empty_mask = sitk.Image(ct_image.GetSize(), sitk.sitkUInt8)
        empty_mask.CopyInformation(ct_image)

        empty_mask_path = output_dir_gts / f"{case_name}.nii.gz"
        sitk.WriteImage(empty_mask, str(empty_mask_path))

        if verbose:
            print(f"Empty GT mask saved at {shorten_path(empty_mask_path)} with shape {ct_image.GetSize()}")
        return
    
    if verbose:
        print(f"For RTSTRUCT conversion, using CT from {shorten_path(ct_dcm)} and RTSTRUCT from {shorten_path(rt_dcm)}")

    plastimatch_rtstruct_to_nifti(
        ct_dcm=ct_dcm,
        rt_dcm=rt_dcm,
        output_dir_struct=output_dir_struct,
        rename_map=None
    )

    # Extract total_tumor_burden mask
    gt_files = list(output_dir_struct.glob("*total_tumor_burden*.nii.gz"))
    if not gt_files:
        print(f"Warning: No total_tumor_burden mask found for {case_name}")
        return
    if len(gt_files) > 1:
        print(f"Warning: Multiple total_tumor_burden masks found for {case_name}. Using the first one.")

    # output_dir_ttb.mkdir(exist_ok=True, parents=True)
    gt_dest = output_dir_gts / f"{case_name}.nii.gz"
    shutil.copy(gt_files[0], gt_dest)

    if verbose:
        print(f"Copied GT mask {gt_files[0].name} to {shorten_path(gt_dest)}")

def _resample_ct_to_pet(ct_path, pt_path, verbose):
    ct_img = sitk.ReadImage(ct_path)
    pt_img = sitk.ReadImage(pt_path)
    resampled_ct = sitk.Resample(ct_img, pt_img)
    sitk.WriteImage(resampled_ct, ct_path)

    # Confirm resampling
    resampled_ct = sitk.ReadImage(ct_path)
    if resampled_ct.GetSize() != pt_img.GetSize():
        raise RuntimeError(f"Resampling CT ({shorten_path(ct_path)}) to PET ({shorten_path(pt_path)}) failed: CT size {resampled_ct.GetSize()} does not match PET size {pt_img.GetSize()}")

    if verbose:
        print(f"Resampled CT saved at {shorten_path(ct_path)} with shape {resampled_ct.GetSize()} to match PET shape {pt_img.GetSize()}")
        print(f"PET path: {shorten_path(pt_path)}")