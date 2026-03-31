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
import csv

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

    # if verbose:
    print(f"Resampled CT saved at {shorten_path(ct_path)} with shape {resampled_ct.GetSize()} to match PET shape {pt_img.GetSize()}")
        # print(f"PET path: {shorten_path(pt_path)}")

# MAIN PRE-PROCESSING FUNCTION

def pre_process(input_path, 
                input_ct,
                input_pet,
                rtstruct_processing, 
                output_pred_dir,
                output_prepro_dir,
                handling_dicoms,
                handling_subdir_niftis,
                handling_flattened_niftis,
                handling_direct_niftis,
                preprocess_only,
                verbose, 
                overwrite):
    ct_dicom_case_map = {} # Map of case names to CT DICOM directories (only for DICOM input)

    if handling_dicoms or handling_subdir_niftis:
        case_dirs = [d for d in input_path.iterdir() if d.is_dir()] if input_path is not None else []
        if not case_dirs:
            raise ValueError(f"No case directories found in {input_path}")

        output_dir_structs = (input_path.parent / f"{input_path.name}_gt_lesion_segmentations") if rtstruct_processing else None
        output_dir_gts = (input_path.parent / f"{input_path.name}_gt_ttb_segmentations") if rtstruct_processing else None

        # Unified logic for both DICOM and NIfTI subdir cases
        not_preprocessed_case_dirs, list_of_lists_already_prepro, ct_dicom_case_map = find_preprocessed(
            case_dirs,
            output_prepro_dir,
            rtstruct_processing,
            output_dir_gts,
            handling_dicoms,
            verbose
        )
        # print(f"not_preprocessed_case_dirs: {not_preprocessed_case_dirs}; list_of_lists_already_prepro: {list_of_lists_already_prepro}; ct_dicom_case_map: {ct_dicom_case_map}")
        print(f"Cases needing preprocessing ({len(not_preprocessed_case_dirs)}): {[d.name for d in not_preprocessed_case_dirs]}")
        print(f"Cases already preprocessed ({len(list_of_lists_already_prepro)}): {[Path(p[0]).stem.rsplit('_000', 1)[0] for p in list_of_lists_already_prepro]}")

        # 3. For post-processing, use all preprocessed cases
        if not_preprocessed_case_dirs:
            # Only process cases not yet preprocessed
            if handling_dicoms:
                list_of_lists_newly_prepro = handle_dicoms(
                                                not_preprocessed_case_dirs,
                                                output_prepro_dir,
                                                rtstruct_processing,
                                                output_dir_structs,
                                                output_dir_gts,
                                                verbose,
                                                overwrite,
                                                delete_structs_dir=False)
            elif handling_subdir_niftis:
                list_of_lists_newly_prepro = handle_subdir_niftis(not_preprocessed_case_dirs,
                                                            output_prepro_dir,
                                                            rtstruct_processing,
                                                            output_dir_gts,
                                                            verbose)
        else:
            list_of_lists_newly_prepro = []
        # print(f"newly_preprocessed: {list_of_lists_newly_prepro};")
        print(f"Cases newly preprocessed ({len(list_of_lists_newly_prepro)}): {[Path(p[0]).stem.rsplit('_000', 1)[0] for p in list_of_lists_newly_prepro]}")
        list_of_lists_prepro = list_of_lists_already_prepro + list_of_lists_newly_prepro

        # 4. For prediction, process only cases needing prediction
        if preprocess_only:
            if verbose:
                print("Pre-processing only; skipping inference candidate filtering.")
            list_of_lists_pred = []
        elif not overwrite:
            list_of_lists_pred = find_predicted(list_of_lists_prepro, 
                                                output_pred_dir, 
                                                preprocess_only,
                                                verbose=verbose)
        else:
            if verbose:
                print(f"Overwriting enabled. All cases will be considered for inference, even if they have existing predictions in {shorten_path(output_pred_dir)}.")
            list_of_lists_pred = list_of_lists_prepro
        if not list_of_lists_pred:
            print("All cases have existing predictions. Nothing to infer.")
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
        if preprocess_only:
            if verbose:
                print("Pre-processing only; skipping inference candidate filtering.")
            list_of_lists_pred = []
        elif not overwrite:
            list_of_lists_pred = find_predicted(list_of_lists_prepro, 
                                                output_pred_dir,
                                                preprocess_only,
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
        try:
            ct_img = sitk.ReadImage(str(ct_path))
            pt_img = sitk.ReadImage(str(pt_path))
        except Exception as e:
            raise ValueError(f"Error reading NIfTI files: {e}")

        ct_stem = ct_path.stem
        pt_stem = pt_path.stem
        # print(f"Input CT stem: {ct_stem}; PET stem: {pt_stem}")
        if '_0000' in ct_stem and '_0001' in pt_stem:
            ct_base = ct_stem.rsplit('_0000', 1)[0]
            pt_base = pt_stem.rsplit('_0001', 1)[0]
        elif 'CT' in ct_stem and 'PT' in pt_stem:
            ct_base = ct_stem.rsplit('CT', 1)[0]
            pt_base = pt_stem.rsplit('PT', 1)[0]
        # print(f"CT base: {ct_base}; PET base: {pt_base}")

        # Acquire sizes and spacings
        ct_size = ct_img.GetSize()
        pt_size = pt_img.GetSize()
        ct_spacing = ct_img.GetSpacing()
        pt_spacing = pt_img.GetSpacing()
        # Resample if sizes OR SPACINGS differ
        if ct_size != pt_size or ct_spacing != pt_spacing:
            print(f"CT and PET sizes ({ct_size} | {pt_size}) / spacings ({ct_spacing} | {pt_spacing}) do not match for {ct_base}. Resampling required.")
            resample_ct_to_pet(ct_path, pt_path, verbose=True)
        # Always include in preprocessed list
        list_of_lists_prepro = [[str(ct_path), str(pt_path)]]
        # Only include in pred list if prediction does not exist or overwrite is True
        if preprocess_only:
            if verbose:
                print("Pre-processing only; skipping inference candidate filtering.")
            list_of_lists_pred = []
        else:
            pred_file = Path(output_pred_dir) / f"{ct_base}.nii.gz"
            # print(f"Prediction file path: {pred_file}")
            if pred_file.exists() and not overwrite:
                print(f"Skipping {ct_base}: prediction exists at {shorten_path(pred_file)}.")
                list_of_lists_pred = []
            else:
                list_of_lists_pred = [[str(ct_path), str(pt_path)]]
        return list_of_lists_prepro, list_of_lists_pred, ct_dicom_case_map

    else:
        raise ValueError(f"No valid NIfTI or DICOM files found in {input_path if input_path is not None else '[input_ct/input_pet]'}.")


# ANTI-REDUNDANCY FUNCTIONS

def find_preprocessed(case_dirs, 
                        output_prepro_dir, 
                        rtstruct_processing, 
                        output_dir_gts, 
                        handling_dicoms,
                        verbose):

    if not case_dirs:
        print(f"No case directories to check.")
        return []

    not_preprocessed_case_dirs = []
    list_of_lists_already_prepro = []

    ct_dicom_case_map = defaultdict(list) # Map of case names to CT DICOM directories (only for DICOM input)

    for case_dir in tqdm(case_dirs, desc="Checking cases for preprocessing"):
        case_name = case_dir.name
        ct_path = Path(output_prepro_dir) / f"{case_name}_0000.nii.gz"
        pt_path = Path(output_prepro_dir) / f"{case_name}_0001.nii.gz"

        rtstruct_found = False

        # Consolidated DICOM discovery for the case directory (robust to various layouts)
        if handling_dicoms and rtstruct_processing:
            # New get_dicom_dir_and_type returns a dict: modality -> [dir, ...]
            dicom_map = get_dicom_dir_and_type(case_dir)
            if dicom_map:
                # If CT directories were found, take the first one for this case
                ct_dirs = dicom_map.get('CT', [])
                if ct_dirs:
                    if case_name not in ct_dicom_case_map:
                        ct_dicom_case_map[case_name] = ct_dirs[0]
                        if verbose:
                            print(f"Mapped case {case_name} -> CT DICOM dir {shorten_path(ct_dirs[0])}")

                # If any RTSTRUCT directories were found, mark rtstruct_found
                rt_dirs = dicom_map.get('RTSTRUCT', [])
                if rt_dirs:
                    rtstruct_found = True
                    if verbose:
                        print(f"Found RTSTRUCT DICOM directory {shorten_path(rt_dirs[0])} for case {case_name}")
            else:
                if verbose:
                    print(f"No DICOM directories detected for case {case_name}. GT generation will be skipped.")

        ct_done, pt_done, gt_done = confirm_already_preprocessed(
            ct_path, 
            pt_path, 
            output_dir_gts,
            case_name, 
            rtstruct_found,
            verbose
        )

        if rtstruct_found:
            needs_preprocessing = not (ct_done and pt_done and gt_done)
        else:
            needs_preprocessing = not (ct_done and pt_done)

        if needs_preprocessing:
            if verbose:
                print(f"Case {case_name} is NOT fully preprocessed. Will process.")
            not_preprocessed_case_dirs.append(case_dir)
        else:
            if verbose:
                print(f"Case {case_name} is fully preprocessed. Skipping.")
            list_of_lists_already_prepro.append([str(ct_path), str(pt_path)])
                            

    return not_preprocessed_case_dirs, list_of_lists_already_prepro, ct_dicom_case_map

def confirm_already_preprocessed(ct_path, 
                                    pt_path, 
                                    output_dir_gts, 
                                    case_name, 
                                    rtstruct_found,
                                    verbose=False):
    ct_done = ct_path.exists()
    pt_done = pt_path.exists()
    gt_done = False

    # Only check shape/spacing if both CT and PT exist
    match = True
    if ct_done and pt_done:
        try:
            ct_img = sitk.ReadImage(str(ct_path))
            pt_img = sitk.ReadImage(str(pt_path))
            ct_size = ct_img.GetSize()
            pt_size = pt_img.GetSize()
            ct_spacing = ct_img.GetSpacing()
            pt_spacing = pt_img.GetSpacing()
            if verbose:
                print(f"[INFO] {case_name}: CT size {ct_size}, spacing {ct_spacing}; PT size {pt_size}, spacing {pt_spacing}")
            if ct_size != pt_size or ct_spacing != pt_spacing:
                match = False
                print(f"[Shape/Spacing Mismatch] {case_name}: CT size {ct_size}, spacing {ct_spacing} != PT size {pt_size}, spacing {pt_spacing}")
                
        except Exception as e:
            match = False
            if verbose:
                print(f"[Error] Could not read CT/PT for shape/spacing check: {e}")
    if ct_done and pt_done and not match:
        ct_done = pt_done = False

    if rtstruct_found: # Only check GT if RTSTRUCT processing is enabled and RTSTRUCT was found
        gt_file = output_dir_gts / f"{case_name}.nii.gz"
        gt_done = gt_file.exists() #and any(gt_dir.glob("*.nii.gz"))
        if verbose:
            print(f"Checking GT at {shorten_path(gt_file)}: exists={gt_file.exists()}, files_found={gt_done}")

    if verbose:
        print(f"Preprocessing check for {case_name}: CT={ct_done}, PT={pt_done}, GT={gt_done}")

    return ct_done, pt_done, gt_done

def find_predicted(list_of_lists_prepro, 
                   output_pred_dir, 
                   preprocess_only,
                   verbose=True):
    """
    Given list_of_lists_prepro ([[ct_path, pt_path], ...]), return only those pairs for which prediction does not exist.
    """
    if preprocess_only is None:
        if verbose:
            print("Pre-processing only; skipping inference candidate filtering.")
        return []

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

def handle_dicoms(not_preprocessed_case_dirs,
                    output_prepro_dir,
                    rtstruct_processing, 
                    output_dir_structs, 
                    output_dir_gts, 
                    verbose, 
                    overwrite, 
                    delete_structs_dir):
    # Make dir to store pre-processed NIfTIs in
    case_dict = defaultdict(list)

    if not not_preprocessed_case_dirs:
        print(f"No case directories found. Nothing to process.")
        return []

    for case_dir in tqdm(not_preprocessed_case_dirs, desc="Pre-processing DICOM cases"):
        case_name = case_dir.name
        print(f"Processing case: {case_name}")
        if verbose:
            print("="*60)
            print(f"\nCase: {case_name}")
            print("="*60)

        for study_dir in case_dir.iterdir():
            if not study_dir.is_dir():
                print(f"Skipping {shorten_path(study_dir)}: not a directory.")
                continue

            dicom_series = get_modality_dirs_and_validate_pet(study_dir,
                                                                output_dir=output_prepro_dir,
                                                                verbose=verbose)
            if dicom_series is None:
                continue

            ct_path = Path(f"{output_prepro_dir}/{case_name}_0000.nii.gz")
            pt_path = Path(f"{output_prepro_dir}/{case_name}_0001.nii.gz")
            sizes = {}
            spacings = {}

            if 'CT' in dicom_series:
                process_dicom(dicom_series, 'CT', ct_path, sizes, spacings, verbose)

            if 'PT' in dicom_series:
                process_dicom(dicom_series, 'PT', pt_path, sizes, spacings, verbose)

            if rtstruct_processing:
                process_rtstruct(dicom_series, study_dir, case_name, output_dir_structs, output_dir_gts, verbose)

            if 'CT' in sizes and 'PT' in sizes and (sizes['CT'] != sizes['PT']) and (spacings['CT'] != spacings['PT']):
                print(f"WARNING: CT and PET sizes/spacings ({sizes['CT']} | {sizes['PT']})/({spacings['CT']} | {spacings['PT']}) do not match for {case_name}. Resampling required.")
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
        return volume.GetSize(), volume.GetSpacing()

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
        return volume_sitk.GetSize(), volume_sitk.GetSpacing()

def process_dicom(dicom_series, 
                    modality, 
                    nifti_path, 
                    sizes,
                    spacings,
                    verbose
                ):
    dicom_dir = dicom_series[modality][0]
    ds = dicom_series[modality][1]
    is_pet = modality == "PT"

    size, spacing = dicom_to_nifti(dicom_dir,
                            nifti_path, 
                            is_pet, 
                            ds, 
                            verbose)
    sizes[modality] = size
    spacings[modality] = spacing


# RTSTRUCT HANDLING

def extract_dir_distinguisher(rtstruct_dirs):
    """
    Given a list of RTSTRUCT directory paths, extract the distinguishing characters
    from their names (not the .dcm file names, but the parent dir names).
    
    For example:
      - /path/RTSTRUCT_1/ -> distinguisher: "1"
      - /path/RTSTRUCT_2/ -> distinguisher: "2"
      - /path/RTSTRUCT_ROI_v1/ -> distinguisher: "ROI_v1"
    
    Returns a dict: {dir_path: distinguisher_string}
    """
    if len(rtstruct_dirs) <= 1:
        return {rtstruct_dirs[0]: ""} if rtstruct_dirs else {}
    
    dir_names = [Path(d).name for d in rtstruct_dirs]
    
    # Find the common prefix and suffix
    common_prefix_len = 0
    for i in range(min(len(name) for name in dir_names)):
        if all(name[i] == dir_names[0][i] for name in dir_names):
            common_prefix_len = i + 1
        else:
            break
    
    common_suffix_len = 0
    for i in range(1, min(len(name) for name in dir_names) + 1):
        if all(name[-i] == dir_names[0][-i] for name in dir_names):
            common_suffix_len = i
        else:
            break
    
    # Extract distinguisher for each directory
    distinguishers = {}
    for dir_path, dir_name in zip(rtstruct_dirs, dir_names):
        if common_suffix_len > 0:
            distinguisher = dir_name[common_prefix_len:-common_suffix_len]
        else:
            distinguisher = dir_name[common_prefix_len:]
        
        # If still empty, use the full name as fallback
        if not distinguisher:
            distinguisher = dir_name
        
        distinguishers[dir_path] = distinguisher
    
    return distinguishers

def process_rtstruct(dicom_series, 
                        study_dir, 
                        case_name,
                        output_dir_structs, 
                        output_dir_gts,
                        verbose=False
                        ):
    ct_dir = dicom_series['CT'][0]
    ct_dcm = next(ct_dir.glob("*.dcm"), None)

    rtstruct_data = dicom_series['RTSTRUCT'] if 'RTSTRUCT' in dicom_series else []
    if not rtstruct_data:
        print(f"No RTSTRUCT DICOM directories found for {case_name}. Skipping GT generation.")
        return
    print(f"Found {len(rtstruct_data)} RTSTRUCT DICOM directory(ies) for {case_name}.")

    rtstruct_dirs = []
    # Extract the series of RTSTRUCT directories
    for i in range(len(rtstruct_data)):
        rtstruct_dir = Path(rtstruct_data[i][0])
        print(f"  RTSTRUCT directory {i+1}: {shorten_path(rtstruct_dir)}")
        rtstruct_dirs.append(rtstruct_dir)
    
    # Extract distinguishers for organizing output
    print(f"Processing {len(rtstruct_dirs)} RTSTRUCT(s) for case {case_name}")
    distinguishers = extract_dir_distinguisher(rtstruct_dirs)
    
    # Process each RTSTRUCT directory
    for rt_dir_idx, rt_dir in enumerate(rtstruct_dirs):
        rt_dcm = next(rt_dir.glob("*.dcm"), None)
        
        if rt_dcm is None:
            print(f"No RTSTRUCT DICOM file found in {shorten_path(rt_dir)}. Skipping this RTSTRUCT.")
            continue
        
        # Create output subdirectory based on distinguisher
        distinguisher = distinguishers[rt_dir]
        if distinguisher:
            output_dir_struct = output_dir_structs / case_name / f"rtstruct_{distinguisher}"
        else:
            # Fallback if only one RTSTRUCT
            output_dir_struct = output_dir_structs / case_name
        
        output_dir_structs.mkdir(parents=True, exist_ok=True)
        output_dir_gts.mkdir(parents=True, exist_ok=True)
        output_dir_struct.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f"Processing RTSTRUCT {rt_dir_idx+1}/{len(rtstruct_dirs)} for {case_name}")
            print(f"  Using CT from {shorten_path(ct_dcm)}")
            print(f"  Using RTSTRUCT from {shorten_path(rt_dcm)}")
            print(f"  Output dir: {shorten_path(output_dir_struct)}")
        
        # Convert RTSTRUCT to NIfTI
        plastimatch_rtstruct_to_nifti(
            ct_dcm=ct_dcm,
            ct_dir=ct_dir,
            rt_dcm=rt_dcm,
            output_dir_struct=output_dir_struct,
            rename_map=None
        )

        # Save GT with distinguisher in filename if multiple RTSTRUCTs
        if distinguisher:
            gt_dest = output_dir_gts / f"{case_name}_{distinguisher}.nii.gz"
        else:
            gt_dest = output_dir_gts / f"{case_name}.nii.gz"

        # Discover total_tumor_burden mask (for fallback and reference geometry)
        gt_files = list(output_dir_struct.glob("*total_tumor_burden*.nii.gz"))
        if not gt_files:
            gt_files = list(output_dir_struct.glob("*ttb*.nii.gz"))
        if not gt_files:
            gt_files = list(output_dir_struct.glob("*TTB*.nii.gz"))
        if len(gt_files) > 1:
            print(f"Warning: Multiple total_tumor_burden masks found. Using the first one as reference/fallback.")

        # Prefer creating TTB from individual lesion masks
        lesion_files = sorted(output_dir_struct.glob("*_lesion.nii.gz"))
        if lesion_files:
            if verbose:
                print(f"Found {len(lesion_files)} lesion mask(s) for {case_name}. Combining into TTB mask.")

            lesion_ref_img = sitk.ReadImage(str(lesion_files[0]))

            def _geom_close(img_a, img_b, spacing_atol=1e-4, origin_atol=1e-2, direction_atol=1e-6):
                return (
                    img_a.GetSize() == img_b.GetSize()
                    and np.allclose(img_a.GetSpacing(), img_b.GetSpacing(), atol=spacing_atol)
                    and np.allclose(img_a.GetOrigin(), img_b.GetOrigin(), atol=origin_atol)
                    and np.allclose(img_a.GetDirection(), img_b.GetDirection(), atol=direction_atol)
                )

            # Use TTB reference only when it is geometrically consistent with lesion masks.
            # Otherwise use lesion grid to avoid slice fragmentation from coarse/misaligned resampling.
            if gt_files:
                ttb_ref_img = sitk.ReadImage(str(gt_files[0]))
                if _geom_close(ttb_ref_img, lesion_ref_img):
                    ref_img = ttb_ref_img
                    if verbose:
                        print(f"  Using reference grid from {gt_files[0].name} (geometry matches lesion masks)")
                else:
                    ref_img = lesion_ref_img
                    print(
                        f"Warning: TTB reference geometry differs from lesion masks for {case_name}. "
                        f"Using lesion reference grid ({lesion_files[0].name}) to preserve lesion continuity."
                    )
            else:
                ref_img = lesion_ref_img
                if verbose:
                    print(f"  No TTB file found. Using reference grid from {lesion_files[0].name}")

            combined_arr = np.zeros_like(sitk.GetArrayFromImage(ref_img), dtype=bool)

            for lesion_file in lesion_files:
                lesion_img = sitk.ReadImage(str(lesion_file))

                # Resample lesion mask to reference grid when geometry differs
                same_geometry = _geom_close(lesion_img, ref_img)
                if not same_geometry:
                    if verbose:
                        print(f"  Resampling {lesion_file.name} to reference geometry")
                    lesion_img = sitk.Resample(
                        lesion_img,
                        ref_img,
                        sitk.Transform(),
                        sitk.sitkNearestNeighbor,
                        0,
                        sitk.sitkUInt8
                    )

                lesion_arr = sitk.GetArrayFromImage(lesion_img) > 0
                combined_arr = np.logical_or(combined_arr, lesion_arr)

            combined_img = sitk.GetImageFromArray(combined_arr.astype(np.uint8))
            combined_img.CopyInformation(ref_img)

            # Conservative robustness step for sparse RTSTRUCT contours:
            # if foreground exists on non-consecutive z-slices, close only along z
            # with a small kernel (~3 mm) to bridge minor missing-slice gaps.
            fg_slices = np.where(np.any(combined_arr, axis=(1, 2)))[0]
            if len(fg_slices) > 1:
                z_gaps = np.diff(fg_slices) - 1
                if np.any(z_gaps > 0):
                    z_spacing_mm = float(ref_img.GetSpacing()[2])
                    z_radius = max(1, int(round(3.0 / max(z_spacing_mm, 1e-6))))
                    z_radius = min(z_radius, 3)  # keep conservative
                    if verbose:
                        print(
                            f"  Detected sparse z-slice occupancy (max missing gap: {int(np.max(z_gaps))} slices). "
                            f"Applying conservative z-only binary closing (radius={z_radius})."
                        )
                    combined_img = sitk.BinaryMorphologicalClosing(
                        combined_img,
                        [0, 0, z_radius],
                        sitk.sitkBall,
                    )
                    combined_img = sitk.Cast(combined_img > 0, sitk.sitkUInt8)

            sitk.WriteImage(combined_img, str(gt_dest))
        else:
            # Fallback: use total_tumor_burden mask if no lesion masks are present
            if not gt_files:
                print(f"Warning: No *_lesion.nii.gz files and no total_tumor_burden mask found for {case_name} from RTSTRUCT {distinguisher}")
                continue
            shutil.copy(gt_files[0], gt_dest)

        if verbose:
            print(f"Copied GT mask to {shorten_path(gt_dest)}")

def plastimatch_rtstruct_to_nifti(ct_dcm, 
                                  ct_dir,
                                  rt_dcm, 
                                  output_dir_struct,
                                  rename_map=None):
    ct_dcm = Path(ct_dcm)
    ct_dir = Path(ct_dir) # Not used currently, but kept for potential future use
    rt_dcm = Path(rt_dcm)

    try:
        command = [
            'plastimatch', 'convert',
            '--input', str(rt_dcm),
            # '--referenced-ct', str(ct_dcm),
            '--referenced-ct', str(ct_dir),
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

# NIfTI HANDLING

def handle_subdir_niftis(case_dirs, 
                        output_prepro_dir, 
                        rtstruct_processing, 
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
                                        case_name, rtstruct_processing, verbose
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

    for f in tqdm(nii_files, desc="Processing flattened NIfTI files"):
        if verbose:
            print(f"Processing {shorten_path(f)}")
        base = f.stem.rsplit('_000', 1)[0]
        nii_path = Path(output_prepro_dir) / f"{base}.nii.gz"

        # Check if ct and pet niftis exist and are the same size
        ct_path = Path(output_prepro_dir) / f"{base}_0000.nii.gz"
        pt_path = Path(output_prepro_dir) / f"{base}_0001.nii.gz"

        if ct_path.exists() and pt_path.exists():
            ct_img = sitk.ReadImage(str(ct_path))
            pt_img = sitk.ReadImage(str(pt_path))
            ct_size = ct_img.GetSize()
            pt_size = pt_img.GetSize()
            ct_spacing = ct_img.GetSpacing()
            pt_spacing = pt_img.GetSpacing()
            if ct_size != pt_size or ct_spacing != pt_spacing:
                if verbose:
                    print(f"CT and PET sizes/spacings ({ct_size} | {pt_size})/({ct_spacing} | {pt_spacing}) do not match for {base}. Resampling required.")
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

def get_modality_dirs_and_validate_pet(study_dir, 
                                        output_dir=None,
                                        verbose=False):
    """
    Checks all DICOM dirs in a study, groups them by modality (CT/PT), and validates PET if found.
    Returns a dict like {'CT': (dir, ds), 'PT': (dir, ds), 'RTSTRUCT': [(dir1, ds1), (dir2, ds2), ...]}
    """
    dicom_series = defaultdict(list)
    # Use a global or external error log if available
    global error_log
    if 'error_log' not in globals():
        error_log = []

    # Get case name for error log
    # case_name = Path(study_dir).name
    # case_name is the parent dir of study_dir
    case_name = Path(study_dir).parent.name

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
            msg = f"Conflicting tomographic modalities in {shorten_path(dicom_dir, 5)}: {tomo_modalities_found}. Skipping entire study."
            print(f"ERROR: {msg}")
            error_log.append({'case': case_name, 'reason': msg})
            return None

        if tomo_modalities_found and non_tomo_modalities:
            msg = f"Tomographic modality {tomo_modalities_found} mixed with non-tomographic {non_tomo_modalities} in {shorten_path(dicom_dir, 5)}. Proceeding."
            print(f"WARNING: {msg}")
            # Not a skip, just a warning

        if representative_ds is not None:
            modality = representative_ds.Modality.upper()
        else:
            modality = None

        if modality == "PT":
            # ... existing PET validation code ...
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
                msg = f"Invalid PET series found in {shorten_path(dicom_dir, 5)}. Missing or incorrect: {missing_tags}. Skipping entire study."
                print(msg)
                error_log.append({'case': case_name, 'reason': msg})
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
                msg = f"SimpleITK header read failed in {shorten_path(dicom_dir, 5)}. Skipping entire study. Error: {str(e)}"
                print(msg)
                error_log.append({'case': case_name, 'reason': msg})
                return None

        # Store modalities: CT/PT as single tuple, RTSTRUCT as list of tuples
        if modality is not None:
            if modality == "RTSTRUCT":
                # Append to list for RTSTRUCT (supports multiple)
                dicom_series[modality].append((dicom_dir, representative_ds))
            else:
                # Store as single tuple for CT/PT (overwrite if duplicate, keep largest)
                dicom_series[modality] = (dicom_dir, representative_ds)

    if not {'CT', 'PT'}.issubset(dicom_series.keys()):
        msg = f"CT and/or PET series not found in {shorten_path(study_dir)}. Skipping entire study."
        print(msg)
        error_log.append({'case': case_name, 'reason': msg})
        return None

    # Save error log if output_dir is provided and there are errors
    if output_dir is not None and error_log:
        parent_dir = Path(output_dir).parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        error_log_path = parent_dir / "preprocessing_error_log.csv"
        with open(error_log_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["case", "reason"])
            writer.writeheader()
            for row in error_log:
                writer.writerow(row)
        print(f"[INFO] Error log saved to {error_log_path}")
    
    return dict(dicom_series)

def get_dicom_dir_and_type(dir):
    p = Path(dir)
    dicom_map = defaultdict(list)  # modality -> list of directory paths (strings)

    try:
        entries = list(p.iterdir())
    except Exception as e:
        print(f"get_dicom_dir_and_type: cannot iterate {dir}: {e}")
        return {}

    # Helper to append a directory to the map
    def _add(modality, path):
        if path is None:
            return
        if str(path) not in dicom_map[modality]:
            dicom_map[modality].append(str(path))

    # 1) Top-level files directly inside the case_dir (flat export)
    top_files = [f for f in entries if f.is_file()]
    if top_files:
        for f in top_files[:10]:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True)
                modality = getattr(ds, 'Modality', '').upper()
                if modality in ('CT', 'PT', 'RTSTRUCT'):
                    _add(modality, p)
            except Exception:
                continue

    # 2) Inspect immediate subdirectories (common layout)
    for sub_dir in entries:
        if not sub_dir.is_dir():
            continue
        try:
            sub_entries = list(sub_dir.iterdir())
        except Exception as e:
            print(f"Cannot iterate subdir {shorten_path(sub_dir)}: {e}")
            continue

        # Try a small sample of files first
        sample_files = [f for f in sub_entries if f.is_file()][:10]
        found_modality = None
        for file in sample_files:
            try:
                ds = pydicom.dcmread(file, stop_before_pixels=True)
                modality = getattr(ds, 'Modality', '').upper()
                if modality in ('CT', 'PT', 'RTSTRUCT'):
                    _add(modality, sub_dir)
                    found_modality = modality
                    # For RTSTRUCT keep ALL directories (don't filter to just one)
                    # For CT/PT we can break from sample
                    if modality in ('CT', 'PT'):
                        break
            except Exception:
                continue

        # If sample didn't find CT/PT, fall back to a full scan of this subdir
        if found_modality not in ('CT', 'PT'):
            for file in sub_entries:
                if not file.is_file():
                    continue
                try:
                    ds = pydicom.dcmread(file, stop_before_pixels=True)
                    modality = getattr(ds, 'Modality', '').upper()
                    if modality in ('CT', 'PT', 'RTSTRUCT'):
                        _add(modality, sub_dir)
                        # If CT or PT found, we can stop scanning this subdir
                        if modality in ('CT', 'PT'):
                            break
                except Exception:
                    continue

        # 3) If this subdir itself contains directory(ies) (common layout: case_dir/dicom_dir/CT_dir),
        # inspect those child directories one level deeper. We keep this shallow (depth=2) to
        # avoid expensive recursive scans while covering the usual PACS export layouts.
        try:
            child_dirs = [d for d in sub_entries if d.is_dir()]
        except Exception:
            child_dirs = []

        for child in child_dirs:
            # If we've already recorded CT/PT for this child (as sub_dir), skip
            # but still allow RTSTRUCTs to accumulate from multiple places
            # Sample a few files first
            try:
                child_entries = list(child.iterdir())
            except Exception as e:
                print(f"Cannot iterate nested subdir {shorten_path(child)}: {e}")
                continue

            sample_files_child = [f for f in child_entries if f.is_file()][:10]
            found_mod_child = None
            for file in sample_files_child:
                try:
                    ds = pydicom.dcmread(file, stop_before_pixels=True)
                    modality = getattr(ds, 'Modality', '').upper()
                    if modality in ('CT', 'PT', 'RTSTRUCT'):
                        _add(modality, child)
                        found_mod_child = modality
                        if modality in ('CT', 'PT'):
                            break
                except Exception:
                    continue

            if found_mod_child not in ('CT', 'PT'):
                for file in child_entries:
                    if not file.is_file():
                        continue
                    try:
                        ds = pydicom.dcmread(file, stop_before_pixels=True)
                        modality = getattr(ds, 'Modality', '').upper()
                        if modality in ('CT', 'PT', 'RTSTRUCT'):
                            _add(modality, child)
                            if modality in ('CT', 'PT'):
                                break
                    except Exception:
                        continue

    # Print a summary of what was found for the given case dir
    if dicom_map:
        # Re-order modality lists so the directory with the most files is first.
        # BUT: for RTSTRUCT, keep ALL directories (don't filter to just one)
        for modality, dirs in list(dicom_map.items()):
            if len(dirs) > 1 and modality != 'RTSTRUCT':  # Only filter non-RTSTRUCT
                counts = []
                for d in dirs:
                    try:
                        dpath = Path(d)
                        cnt = sum(1 for f in dpath.iterdir() if f.is_file() and not f.name.startswith('.'))
                    except Exception:
                        cnt = 0
                    counts.append((d, cnt))
                # sort descending by count
                counts.sort(key=lambda x: x[1], reverse=True)
                # replace list with sorted dirs (keeps only largest)
                dicom_map[modality] = [c[0] for c in counts]
                # user-visible message indicating choice
                best_dir, best_count = counts[0]
                try:
                    other_dirs = [shorten_path(Path(x)) for x in dicom_map[modality][1:]]
                except Exception:
                    other_dirs = []
                print(f"[INFO] Multiple {modality} dirs found for {shorten_path(p)}. Choosing largest ({shorten_path(best_dir)}) with {best_count} files. Other dirs: {other_dirs}")
            elif modality == 'RTSTRUCT' and len(dirs) > 1:
                # For RTSTRUCT, keep all directories and print info
                print(f"[INFO] Multiple RTSTRUCT dirs found for {shorten_path(p)}: {[shorten_path(Path(x)) for x in dirs]}")

        summary = {k: [shorten_path(Path(p)) for p in v] for k, v in dicom_map.items()}
        # print(f"[DEBUG] DICOM mapping for {shorten_path(p)}: {summary}")
    else:
        print(f"[WARNING] No CT/PT/RTSTRUCT found in {shorten_path(p)}")

    return dict(dicom_map)

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