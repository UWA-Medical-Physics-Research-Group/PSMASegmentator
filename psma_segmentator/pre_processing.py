""" 
PSMASegmentator is a tool for the automatic segmentation of PSMA 
PET/CT images using Deep Learning (DL).
Copyright (C) 2025 UWA Medical Physics Research Group, The University 
of Western Australia, Crawley, WA, Australia

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

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
import matplotlib.pyplot as plt

# SUPPLEMENTARY FUNCTIONS

def shorten_path(path, max_parts=2):
    path = Path(path)
    parts = path.parts
    if len(parts) <= max_parts * 2 + 1:
        return str(path)

    start = Path(*parts[:max_parts])
    end = Path(*parts[-max_parts:])
    return str(start / "..." / end)

def save_middle_slice_png(np_volume, nifti_path, 
                            tag="pre", is_pet=False,
                            verbose=False):
    """
    Saves the middle axial slice of a 3D numpy volume as a PNG image
    to the parent directory of `nifti_path`, with a tag indicating the processing stage.
    """
    if np_volume.ndim != 3:
        print(f"[ERROR] Volume is not 3D. Shape: {np_volume.shape}")
        return

    mid_slice_idx = np_volume.shape[0] // 2
    mid_slice = np_volume[mid_slice_idx]

    nifti_name = os.path.basename(nifti_path)
    nifti_name = os.path.splitext(nifti_name)[0] # Remove extension
    out_dir = os.path.dirname(nifti_path)
    out_name = f"{nifti_name}_mid_slice_{tag}"
    if is_pet:
        out_name += "_pet"
    else:
        out_name += "_ct"
    out_name += ".png"
    out_path = os.path.join(out_dir, out_name)

    plt.figure(figsize=(6, 6))
    plt.imshow(mid_slice, cmap='gray')
    plt.title(f"{nifti_name}: Mid Slice ({tag})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"Saved middle slice PNG ({tag}) to: {out_path}")


# MAIN PRE-PROCESSING FUNCTION

def pre_process(input_path, incl_rtstructs, 
                output_pred_dir,
                output_prepro_dir,
                handling_dicom,
                nifti_subdirs,
                verbose, overwrite):

    case_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    print(f"DEBUG pre_process case_dirs: {case_dirs}")
    
    if case_dirs:
        # print(f"Found case directories in input path, assuming DICOM input.")
        # print(f"Case dirs: {case_dirs}")
        
        output_dir_structs = (input_path.parent / f"{input_path.name}_structs") if incl_rtstructs else None
        output_dir_gts = (input_path.parent / f"{input_path.name}_gt_segmentations") if incl_rtstructs else None
        if incl_rtstructs:
            output_dir_structs.mkdir(parents=True, exist_ok=True)
            output_dir_gts.mkdir(parents=True, exist_ok=True)

        # Filter to only cases needing prediction
        if not overwrite:
            case_dirs_to_predict = find_predicted(case_dirs, 
                                                    output_pred_dir, 
                                                    mode='case_dirs', 
                                                    verbose=verbose)
        else:
            case_dirs_to_predict = case_dirs

        if not case_dirs_to_predict:
            print("All cases have existing predictions. Nothing to pre-process or infer.")
            return []
        
        print(f"Cases needing prediction: {case_dirs_to_predict}")

        # Among cases needing prediction, check which are already preprocessed
        if not overwrite:
            not_preprocessed, preprocessed = find_preprocessed(case_dirs_to_predict, 
                                                                output_prepro_dir, 
                                                                incl_rtstructs, 
                                                                output_dir_gts, 
                                                                nifti_subdirs, 
                                                                verbose)
        else:
            not_preprocessed, preprocessed = case_dirs_to_predict, [] # If overwriting, all are considered not preprocessed
            
        # print(f"Not preprocessed cases: {not_preprocessed}")
        # print(f"Already preprocessed cases: {preprocessed}")

        # Handle DICOM input
        if handling_dicom:
            newly_preprocessed = handle_dicoms(not_preprocessed,
                                                output_prepro_dir, 
                                                incl_rtstructs, 
                                                output_dir_structs, 
                                                output_dir_gts, 
                                                verbose, overwrite, 
                                                delete_structs_dir=False)
            return preprocessed + (newly_preprocessed or [])
        elif nifti_subdirs:
            return preprocessed + not_preprocessed
    elif not handling_dicom and not nifti_subdirs:
        # Handle flattened NIfTI input
        nii_files = list(input_path.rglob("*.nii.gz"))
        print(f"nifti_files: {nii_files}")
        if not overwrite: # only check for predicted NIfTI files if not overwriting
            nii_files = find_predicted(nii_files, output_pred_dir,
                                        mode='nii_files', verbose=verbose)
        if nii_files:
            newly_processed = handle_flattened_niftis(nii_files, 
                                                            output_prepro_dir, 
                                                            overwrite, verbose)
            print(f"Newly processed NIfTI files: {newly_processed}")
            return newly_processed # + preprocessed
        else:
            print("All NIfTI output files already exist and/or overwrite = False. Nothing to predict.")
            return []

    raise ValueError(f"No valid NIfTI or DICOM files found in {input_path}.")


# ANTI-REDUNDANCY FUNCTIONS

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


def find_preprocessed(case_dirs, 
                        output_prepro_dir, 
                        incl_rtstructs, 
                        output_dir_gts, 
                        nifti_subdirs, 
                        verbose):

    if not case_dirs:
        print(f"No case directories to check.")
        return []

    not_preprocessed = []
    preprocessed = []

    for case_dir in case_dirs:
        case_name = case_dir.name
        ct_path = Path(output_prepro_dir) / f"{case_name}_0000.nii.gz"
        pt_path = Path(output_prepro_dir) / f"{case_name}_0001.nii.gz"

        if nifti_subdirs:
            nifti_files = list(case_dir.rglob("*.nii")) + list(case_dir.rglob("*.nii.gz"))

            def find_modality(modality):
                for f in nifti_files:
                    if modality in f.name.lower() and not "resampled" in f.name.lower():
                        return f
                for f in nifti_files:
                    if modality in f.name.lower():
                        return f
                return None

            ct_file = find_modality("ct")
            pt_file = find_modality("pt")

            # Ensure output dir exists
            ct_path.parent.mkdir(parents=True, exist_ok=True)

            if ct_file is not None:
                if not ct_path.exists():
                    if verbose:
                        print(f"Copying CT: {ct_file} → {ct_path}")
                    shutil.copy(ct_file, ct_path)
            if pt_file is not None:
                if not pt_path.exists():
                    if verbose:
                        print(f"Copying PT: {pt_file} → {pt_path}")
                    shutil.copy(pt_file, pt_path)

            # Resample CT to PET if sizes differ [SHOULD SPLIT THIS FUNCTION OUT]
            if ct_file and pt_file:
                ct_img = sitk.ReadImage(str(ct_path))
                pt_img = sitk.ReadImage(str(pt_path))
                if ct_img.GetSize() != pt_img.GetSize():
                    if verbose:
                        print(f"CT and PET sizes ({ct_img.GetSize()} | {pt_img.GetSize()}) do not match for {case_name}. Resampling required.")
                    resample_ct_to_pet(ct_path, pt_path, verbose=True)

        # # Only proceed if both CT and PT exist
        # if not ct_path.exists() or not pt_path.exists():
        #     continue

        ct_done, pt_done, gt_done = already_preprocessed(
            ct_path, pt_path, output_dir_gts,
            case_name, incl_rtstructs, verbose
        )

        if not (ct_done and pt_done and (not incl_rtstructs or gt_done)):
            if verbose:
                print(f"Case {case_name} is NOT fully preprocessed. Will process.")
            if nifti_subdirs:
                not_preprocessed.append([str(ct_path), str(pt_path)])
            else:
                not_preprocessed.append(case_dir) # retain full case_dir for DICOM handling aber der nifti ist alles klar
        else:
            if verbose:
                print(f"Case {case_name} is fully preprocessed. Skipping.")
            if nifti_subdirs:
                preprocessed.append([str(ct_path), str(pt_path)])
            else:
                preprocessed.append(case_dir) # retain full case_dir for DICOM handling aber der nifti ist alles klar mein bruder

    return not_preprocessed, preprocessed


def already_preprocessed(ct_path, pt_path, output_dir_gts, 
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


# DICOM HANDLING

def handle_dicoms(case_dirs,
                    # input_path, 
                    output_prepro_dir,
                    incl_rtstructs, output_dir_structs, output_dir_gts, 
                    verbose, overwrite, delete_structs_dir):
    # Make dir to store pre-processed NIfTIs in
    os.makedirs(output_prepro_dir, exist_ok=True)
    case_dict = defaultdict(list)

    # case_dirs = [d for d in input_path.iterdir() if d.is_dir()]

    # print(f"case_dirs: {case_dirs}")

    if not case_dirs:
        print(f"No case directories found. Nothing to process.")
        return []

    for case_dir in tqdm(case_dirs, desc="Pre-processing cases"):
        print(f"Processing case directory: {case_dir}")
        case_name = case_dir.name
        if verbose:
            print("="*60)
            print(f"\n\nCase: {case_name}")
            print("="*60)

        for study_dir in case_dir.iterdir():
            if not study_dir.is_dir():
                print(f"Skipping {shorten_path(study_dir)}: not a directory.")
                continue

            dicom_series = get_modality_dirs_and_validate_pet(study_dir, verbose)
            if dicom_series is None:  # the study failed the check
                # Error message already printed in get_modality_dirs_and_validate_pet
                continue

            ct_path, pt_path = Path(f"{output_prepro_dir}/{case_name}_0000.nii.gz"), Path(f"{output_prepro_dir}/{case_name}_0001.nii.gz")
            # Regardless of further processing, add paths to case_dict for later inference

            ct_done, pt_done, gt_done = already_preprocessed(ct_path, pt_path, output_dir_gts, 
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
                process_rtstruct(dicom_series, study_dir, case_name, 
                                    output_dir_structs, output_dir_gts, verbose)

            if 'CT' in sizes and 'PT' in sizes and sizes['CT'] != sizes['PT']:
                if verbose:
                    print(f"CT and PET sizes ({sizes['CT']} | {sizes['PT']}) do not match for {case_name}. Resampling required.")
                resample_ct_to_pet(ct_path, pt_path, verbose)

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


def get_sorted_dicom_files_by_z(dicom_dir):
    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir)
                    if not f.startswith('.') and os.path.isfile(os.path.join(dicom_dir, f))]

    dicoms = []
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            z = float(ds.ImagePositionPatient[2])
            dicoms.append((z, f))
        except Exception as e:
            print(f"[WARNING] Skipping {f}: {e}")

    dicoms.sort(key=lambda x: x[0])
    return [f for (_, f) in dicoms]

def dicom_to_nifti(dicom_dir: str, nifti_path: str, is_pet: bool, ds_for_suv):
    print(f"[INFO] Reading DICOM from: {dicom_dir}")

    if is_pet:
        # --- PET: Use SimpleITK ---
        reader = sitk.ImageSeriesReader()
        dicom_series = reader.GetGDCMSeriesFileNames(str(dicom_dir))
        reader.SetFileNames(dicom_series)
        volume = reader.Execute()
        volume = sitk.DICOMOrient(volume, "LPS")

        suv_factor = get_suv_bw_scale_factor(ds_for_suv)
        print(f"[DEBUG] Applying SUV scale factor: {suv_factor}")
        volume *= suv_factor

        sitk.WriteImage(volume, nifti_path)
        print(f"[INFO] Wrote PET NIfTI to: {nifti_path}")
        return volume.GetSize()

    else:
        # --- CT: Manual stacking with HU rescaling and direction fix ---
        dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir)
                        if not f.startswith('.') and os.path.isfile(os.path.join(dicom_dir, f))]

        dicoms = []
        for f in dicom_files:
            try:
                ds = pydicom.dcmread(f)
                z = float(ds.ImagePositionPatient[2])
                dicoms.append((z, ds))
            except Exception as e:
                print(f"[WARNING] Skipping {f}: {e}")

        dicoms.sort(key=lambda x: x[0])
        sorted_ds = [d[1] for d in dicoms]
        print(f"[DEBUG] Sorted {len(sorted_ds)} CT slices by Z")

        # Apply HU rescale
        hu_slices = []
        for ds in sorted_ds:
            slope = float(getattr(ds, "RescaleSlope", 1))
            intercept = float(getattr(ds, "RescaleIntercept", 0))
            hu = ds.pixel_array.astype(np.float32) * slope + intercept
            hu_slices.append(hu)

        volume_np = np.stack(hu_slices, axis=0)

        # Spatial metadata
        first = sorted_ds[0]
        pixel_spacing = [float(x) for x in first.PixelSpacing]  # [row, col]
        origin = [float(x) for x in first.ImagePositionPatient]

        # Get actual inter-slice spacing from z differences
        z_positions = [float(ds.ImagePositionPatient[2]) for ds in sorted_ds]
        z_diffs = np.diff(z_positions)
        if not np.allclose(z_diffs, z_diffs[0]):
            print(f"[WARNING] Non-uniform slice spacing detected. Using mean: {np.mean(z_diffs)}")
        slice_spacing = float(np.mean(z_diffs)) if len(z_diffs) > 0 else 1.0
        spacing = [slice_spacing] + pixel_spacing  # (z, y, x)

        # Build direction matrix from ImageOrientationPatient
        iop = [float(v) for v in first.ImageOrientationPatient]  # 6 values
        row = np.array(iop[0:3])
        col = np.array(iop[3:6])
        slice_dir = np.cross(row, col)
        direction = np.concatenate([row, col, slice_dir])  # 9 values

        # Convert to SimpleITK
        volume_sitk = sitk.GetImageFromArray(volume_np)  # shape (z, y, x)
        volume_sitk.SetSpacing(spacing[::-1])            # (x, y, z)
        volume_sitk.SetOrigin(origin)
        volume_sitk.SetDirection(direction.tolist())

        sitk.WriteImage(volume_sitk, nifti_path)
        print(f"[INFO] Wrote CT NIfTI to: {nifti_path}")
        return volume_sitk.GetSize()


def dicom_to_nifti(dicom_dir: str, nifti_path: str,
                    is_pet: bool, ds):
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    reader.SetFileNames(dicom_series)
    volume = reader.Execute()
    volume = sitk.DICOMOrient(volume, "LPS")
    size = volume.GetSize() #GetWise

    if is_pet:
        suv_factor = get_suv_bw_scale_factor(ds)
        # Convert the PET image to SUV
        volume *= suv_factor

    sitk.WriteImage(volume, nifti_path)
    return size


def process_dicom(dicom_series, modality, 
                    nifti_path, sizes):
    dicom_dir = dicom_series[modality][0]
    ds = dicom_series[modality][1]
    is_pet = modality == "PT"
    
    size = dicom_to_nifti(dicom_dir, nifti_path, is_pet, ds)
    sizes[modality] = size


# RTSTRUCT HANDLING

def process_rtstruct(dicom_series, study_dir, case_name,
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

def plastimatch_rtstruct_to_nifti(ct_dcm, rt_dcm, output_dir_struct,
                                    rename_map=None):
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


def handle_flattened_niftis(nii_files, output_prepro_dir, 
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
                resample_ct_to_pet(ct_path, pt_path, verbose=True)
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


def resample_ct_to_pet(ct_path, pt_path, verbose):
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


# MODALITY AND PET VALIDATION

def get_modality_dirs_and_validate_pet(study_dir, verbose):
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

        ext_missing = False

        for file in dicom_dir.iterdir():  # dicom_dir.glob("*.dcm"):
            if not file.is_file():
                continue
            if not file.name.endswith(".dcm"):
                ext_missing = True
            try:
                ds = pydicom.dcmread(file, stop_before_pixels=True)
                modality = ds.Modality.upper()
                modalities_present.add(modality)
                if representative_ds is None:
                    representative_ds = ds  # Just one is enough
            except Exception:
                continue
        if ext_missing and verbose:
            print(f"WARNING: DICOM files in {shorten_path(dicom_dir)} do not have '.dcm' extension. This may cause issues.")

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