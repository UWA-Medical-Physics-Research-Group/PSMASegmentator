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
import pyplastimatch
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

# MAIN PRE-PROCESSING FUNCTION

def pre_process(input_path, 
                input_ct,
                input_pet,
                incl_rtstructs, 
                output_pred_dir,
                output_prepro_dir,
                handling_dicoms,
                handling_subdir_niftis,
                handling_flattened_niftis,
                handling_direct_niftis,
                verbose, 
                overwrite):
    ct_dicom_case_map = {} # Map of case names to CT DICOM directories (only for DICOM input)

    if handling_dicoms or handling_subdir_niftis:
        case_dirs = [d for d in input_path.iterdir() if d.is_dir()] if input_path is not None else []
        if not case_dirs:
            raise ValueError(f"No case directories found in {input_path}")

        output_dir_structs = (input_path.parent / f"{input_path.name}_structs") if incl_rtstructs else None
        output_dir_gts = (input_path.parent / f"{input_path.name}_gt_segmentations") if incl_rtstructs else None
        if incl_rtstructs:
            output_dir_structs.mkdir(parents=True, exist_ok=True)
            output_dir_gts.mkdir(parents=True, exist_ok=True)

        # Unified logic for both DICOM and NIfTI subdir cases
        not_preprocessed_case_dirs, list_of_lists_already_prepro = find_preprocessed(case_dirs,
                                                                output_prepro_dir,
                                                                incl_rtstructs,
                                                                output_dir_gts,
                                                                verbose)
        print(f"not_preprocessed_case_dirs: {not_preprocessed_case_dirs}; list_of_lists_already_prepro: {list_of_lists_already_prepro};")

        # 3. For post-processing, use all preprocessed cases
        if not_preprocessed_case_dirs:
            # Only process cases not yet preprocessed
            if handling_dicoms:
                list_of_lists_newly_prepro, ct_dicom_case_map = handle_dicoms(not_preprocessed_case_dirs,
                                                                    output_prepro_dir,
                                                                    incl_rtstructs,
                                                                    output_dir_structs,
                                                                    output_dir_gts,
                                                                    verbose,
                                                                    overwrite,
                                                                    delete_structs_dir=False)
            elif handling_subdir_niftis:
                list_of_lists_newly_prepro = handle_subdir_niftis(not_preprocessed_case_dirs,
                                                            output_prepro_dir,
                                                            incl_rtstructs,
                                                            output_dir_gts,
                                                            verbose)
        else:
            list_of_lists_newly_prepro = []
        print(f"newly_preprocessed: {list_of_lists_newly_prepro};")
        list_of_lists_prepro = list_of_lists_already_prepro + list_of_lists_newly_prepro

        # 4. For prediction, process only cases needing prediction
        if not overwrite:
            list_of_lists_pred = find_predicted(list_of_lists_prepro, 
                                                output_pred_dir, 
                                                verbose=verbose)
        else:
            if verbose:
                print(f"Overwriting enabled. All cases will be considered for inference, even if they have existing predictions in {shorten_path(output_pred_dir)}.")
            list_of_lists_pred = list_of_lists_prepro
        if not list_of_lists_pred:
            print("All cases have existing predictions. Nothing to infer.")
            print(f"CT DICOM case map: {ct_dicom_case_map}")
        return list_of_lists_prepro, list_of_lists_pred, ct_dicom_case_map

    elif handling_flattened_niftis:
        # Handle flattened NIfTI input
        nii_files = list(input_path.rglob("*.nii.gz"))
        # Find all preprocessed NIfTI pairs
        list_of_lists_prepro = handle_flattened_niftis(nii_files, 
                                                        output_prepro_dir, 
                                                        overwrite, 
                                                        verbose)
        # Find NIfTI files needing prediction
        if not overwrite:
            list_of_lists_pred = find_predicted(list_of_lists_prepro, 
                                                output_pred_dir,
                                                verbose=verbose)
        else:
            if verbose:
                print(f"Overwriting enabled. All NIfTI files will be considered for pre-processing and inference, even if they have existing predictions in {shorten_path(output_pred_dir)}.")
            list_of_lists_pred = list_of_lists_prepro
        return list_of_lists_prepro, list_of_lists_pred, ct_dicom_case_map
    
    # Handle direct input_ct and input_pet file paths
    elif handling_direct_niftis:
        ct_path = Path(input_ct)
        pt_path = Path(input_pet)
        base = ct_path.stem.rsplit('_000', 1)[0] if '_000' in ct_path.stem else ct_path.stem
        ct_img = sitk.ReadImage(str(ct_path))
        pt_img = sitk.ReadImage(str(pt_path))
        # Check if CT and PET sizes match
        if ct_img.GetSize() != pt_img.GetSize():
            print(f"CT and PET sizes ({ct_img.GetSize()} | {pt_img.GetSize()}) do not match for {base}. Resampling required.")
            resample_ct_to_pet(ct_path, pt_path, verbose=True)
        # Always include in preprocessed list
        list_of_lists_prepro = [[str(ct_path), str(pt_path)]]
        # Only include in pred list if prediction does not exist or overwrite is True
        pred_file = Path(output_pred_dir) / f"{base}.nii.gz"
        if pred_file.exists() and not overwrite:
            print(f"Skipping {base}: prediction exists at {shorten_path(pred_file)}.")
            list_of_lists_pred = []
        else:
            list_of_lists_pred = [[str(ct_path), str(pt_path)]]
        return list_of_lists_prepro, list_of_lists_pred, ct_dicom_case_map

    else:
        raise ValueError(f"No valid NIfTI or DICOM files found in {input_path if input_path is not None else '[input_ct/input_pet]'}.")


# ANTI-REDUNDANCY FUNCTIONS

def find_preprocessed(case_dirs, 
                        output_prepro_dir, 
                        incl_rtstructs, 
                        output_dir_gts, 
                        verbose):

    if not case_dirs:
        print(f"No case directories to check.")
        return []

    not_preprocessed_case_dirs = []
    list_of_lists_already_prepro = []

    for case_dir in case_dirs:
        case_name = case_dir.name
        ct_path = Path(output_prepro_dir) / f"{case_name}_0000.nii.gz"
        pt_path = Path(output_prepro_dir) / f"{case_name}_0001.nii.gz"

        ct_done, pt_done, gt_done = confirm_already_preprocessed(
            ct_path, pt_path, output_dir_gts,
            case_name, incl_rtstructs, verbose
        )

        if not (ct_done and pt_done and (not incl_rtstructs or gt_done)):
            if verbose:
                print(f"Case {case_name} is NOT fully preprocessed. Will process.")
            not_preprocessed_case_dirs.append(case_dir)
        else:
            if verbose:
                print(f"Case {case_name} is fully preprocessed. Skipping.")
            list_of_lists_already_prepro.append([str(ct_path), str(pt_path)])

    return not_preprocessed_case_dirs, list_of_lists_already_prepro

def confirm_already_preprocessed(ct_path, pt_path, output_dir_gts, 
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

def find_predicted(list_of_lists_prepro, output_pred_dir, verbose=True):
    """
    Given list_of_lists_prepro ([[ct_path, pt_path], ...]), return only those pairs for which prediction does not exist.
    """
    list_of_lists_pred = [] # collate pairs STILL NEEDING prediction
    for pair in list_of_lists_prepro:
        ct_path = Path(pair[0])
        case_name = ct_path.stem.rsplit('_000', 1)[0] if '_000' in ct_path.stem else ct_path.stem
        pred_file = Path(output_pred_dir) / f"{case_name}.nii.gz"
        if pred_file.exists():
            if verbose:
                print(f"Skipping {case_name}: prediction exists at {shorten_path(pred_file)}.")
            continue
        list_of_lists_pred.append(pair)
    return list_of_lists_pred

# DICOM HANDLING

def handle_dicoms(case_dirs,
                    output_prepro_dir,
                    incl_rtstructs, 
                    output_dir_structs, 
                    output_dir_gts, 
                    verbose, 
                    overwrite, 
                    delete_structs_dir):
    # Make dir to store pre-processed NIfTIs in
    case_dict = defaultdict(list)
    # Make dir for mapping case names to CT DICOM dirs
    ct_dicom_case_map = defaultdict(list)

    if not case_dirs:
        print(f"No case directories found. Nothing to process.")
        return [], ct_dicom_case_map

    for case_dir in tqdm(case_dirs, desc="Pre-processing DICOM cases"):
        # print(f"Processing case directory: {case_dir}")
        case_name = case_dir.name
        if verbose:
            print("="*60)
            print(f"\nCase: {case_name}")
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

            ct_done, pt_done, gt_done = confirm_already_preprocessed(ct_path, pt_path, output_dir_gts, 
                                                                    case_name, incl_rtstructs, verbose)

            # Always record CT DICOM dir for mapping
            if 'CT' in dicom_series:
                ct_dicom_dir = dicom_series['CT'][0]
                ct_dicom_case_map[case_name] = str(ct_dicom_dir)
                print(f"CT DICOM dir for case {case_name}: {ct_dicom_dir}")

            # Skip if all required parts are done
            if (not overwrite) and (ct_done and pt_done and (not incl_rtstructs or gt_done)):
                print(f"Skipping {case_name} (preprocessed CT, PET{', GT' if incl_rtstructs else ''} found).") #at {shorten_path(output_prepro_dir)}).
                # Add preprocessed paths to case_dict
                case_dict[case_name].extend([str(ct_path), str(pt_path)])
                continue

            sizes = {}

            if 'CT' in dicom_series:
                if not ct_done:
                    process_dicom(dicom_series, 
                                    'CT', 
                                    ct_path, 
                                    sizes,
                                    verbose)

            if 'PT' in dicom_series:
                if not pt_done:
                    process_dicom(dicom_series, 
                                    'PT', 
                                    pt_path, 
                                    sizes,
                                    verbose)

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
    return list_of_lists, ct_dicom_case_map

def dicom_to_nifti(dicom_dir: str,
                    nifti_path: str,
                    is_pet: bool,
                    ds_for_suv,
                    verbose: bool):
    if is_pet:
        if verbose:
            print(f"Converting PET DICOM to NIfTI: {nifti_path}")
        reader = sitk.ImageSeriesReader()
        dicom_series = reader.GetGDCMSeriesFileNames(str(dicom_dir))
        reader.SetFileNames(dicom_series)
        volume = reader.Execute()
        volume = sitk.DICOMOrient(volume, "LPS")

        suv_factor = get_suv_bw_scale_factor(ds_for_suv)
        volume *= suv_factor

        sitk.WriteImage(volume, nifti_path)
        return volume.GetSize()

    else:
        if verbose:
            print(f"[Converting CT DICOM to NIfTI: {nifti_path}")
        # CT Manual stacking with HU rescaling and direction fix
        dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir)
                        if not f.startswith('.') and os.path.isfile(os.path.join(dicom_dir, f))]

        dicoms = []
        for f in dicom_files:
            try:
                ds = pydicom.dcmread(f)
                z = float(ds.ImagePositionPatient[2])
                dicoms.append((z, ds))
            except Exception as e:
                print(f"WARNING: Skipping {f}: {e}")

        dicoms.sort(key=lambda x: x[0])
        sorted_ds = [d[1] for d in dicoms]

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
        if verbose:
            print(f"Wrote CT NIfTI to: {nifti_path}")
        return volume_sitk.GetSize()

def process_dicom(dicom_series, 
                    modality, 
                    nifti_path, 
                    sizes,
                    verbose):
    dicom_dir = dicom_series[modality][0]
    ds = dicom_series[modality][1]
    is_pet = modality == "PT"

    size = dicom_to_nifti(dicom_dir,
                            nifti_path, 
                            is_pet, 
                            ds, 
                            verbose)
    sizes[modality] = size


# RTSTRUCT HANDLING

def process_rtstruct(dicom_series, 
                        study_dir, 
                        case_name,
                        output_dir_structs, 
                        output_dir_gts,
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
        pyplastimatch.convert(
            input=str(rt_dcm),
            referenced_ct=str(ct_dcm),
            output_prefix=str(output_dir_struct) + os.sep,
            prefix_format='nii.gz',
            prune_empty=True
        )
    except Exception as e:
        raise RuntimeError(f"pyplastimatch RTSTRUCT→NIfTI conversion failed: {e}")

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

# NIfTI HANDLING

def handle_subdir_niftis(case_dirs, 
                        output_prepro_dir, 
                        incl_rtstructs, 
                        output_dir_gts, 
                        verbose):
    """
    Handles NIfTI subdir case for find_preprocessed logic.
    """
    case_dict = defaultdict(list)

    for case_dir in tqdm(case_dirs, desc="Pre-processing NIfTI subdir cases"):
        case_name = case_dir.name
        ct_path = Path(output_prepro_dir) / f"{case_name}_0000.nii.gz"
        pt_path = Path(output_prepro_dir) / f"{case_name}_0001.nii.gz"

        nifti_files = list(case_dir.rglob("*.nii")) + list(case_dir.rglob("*.nii.gz"))

        # Improved CT/PT identification
        ct_candidates = []
        pt_candidates = []
        for f in nifti_files:
            fname = f.name.lower()
            # CT candidates
            if (fname.endswith("_0000.nii.gz") or fname.endswith("_0000.nii")) or ("ct" in fname and not "resampled" in fname):
                ct_candidates.append(f)
            # PT candidates
            if (fname.endswith("_0001.nii.gz") or fname.endswith("_0001.nii")) or (("pt" in fname or "pet" in fname) and not "resampled" in fname):
                pt_candidates.append(f)

        # Prefer most specific match
        ct_file = ct_candidates[0] if ct_candidates else None
        pt_file = pt_candidates[0] if pt_candidates else None
        print(f"Identified CT: {ct_file}, PET: {pt_file}")

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

        # Resample CT to PET if sizes differ
        if ct_file and pt_file:
            ct_img = sitk.ReadImage(str(ct_path))
            pt_img = sitk.ReadImage(str(pt_path))
            if ct_img.GetSize() != pt_img.GetSize():
                if verbose:
                    print(f"CT and PET sizes ({ct_img.GetSize()} | {pt_img.GetSize()}) do not match for {case_name}. Resampling required.")
                resample_ct_to_pet(ct_path, pt_path, verbose=True)

        ct_done, pt_done, gt_done = confirm_already_preprocessed(
                                        ct_path, pt_path, output_dir_gts,
                                        case_name, incl_rtstructs, verbose
                                    )

        # Only add if both CT and PT are present and exist on disk
        if ct_file and pt_file and ct_path.exists() and pt_path.exists():
            case_dict[case_name].extend([str(ct_path), str(pt_path)])

    list_of_lists = [sorted(files) for files in case_dict.values() if len(files) == 2]
    if not list_of_lists:
        print(f"No valid CT/PET NIfTI pairs collated from filtered case_dirs. Nothing to process.")
    return list_of_lists

def handle_flattened_niftis(nii_files,
                            output_prepro_dir,
                            overwrite, 
                            verbose):
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

    list_of_lists_prepro = [sorted(files) for files in case_dict.values()]
    if not list_of_lists_prepro:
        print(f"All pre-processed NIfTIs already exist in {output_prepro_dir}. Nothing to process.")
    return list_of_lists_prepro


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