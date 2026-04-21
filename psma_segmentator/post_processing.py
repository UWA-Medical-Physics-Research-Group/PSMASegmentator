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

NB: This software is intended for RESEARCH PURPOSES ONLY.
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*", module=".*TotalSegmentator.*")


import SimpleITK as sitk
import os
import numpy as np
import totalsegmentator
import time
from scipy.ndimage import label, generate_binary_structure
from collections import defaultdict
from tqdm import tqdm
import json
import logging
from pathlib import Path
import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt
import math
from nibabel.orientations import aff2axcodes, axcodes2ornt, ornt_transform, apply_orientation
from nibabel.processing import resample_from_to
from scipy.ndimage import zoom
import torch
import torch.nn.functional as F
from monai.networks.nets import DenseNet121
from monai.transforms import Compose, NormalizeIntensity
import csv
from datetime import datetime
import time
import subprocess
import pydicom
from sklearn.mixture import GaussianMixture


from psma_segmentator.pre_processing import shorten_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# IMPORTANT VARS

TARGET_SPACING = (4.0728, 4.0728, 2.0)
TARGET_ORIENT = ('L', 'A', 'S')

## Define the mapping from TotalSegmentator labels to biomarker regions ##
biomarker_regions = ['whole_body', 'bone', 'prostate', 'nodal_below_cib', 'nodal_above_cib', 'visceral']
totalseg_labels_and_regions = {
    "whole_body": { # same as 'total' in TS mapping except for CIB labels (200, 201)
        "1": "spleen",
        "2": "kidney_right",
        "3": "kidney_left",
        "4": "gallbladder",
        "5": "liver",
        "6": "stomach",
        "7": "pancreas",
        "8": "adrenal_gland_right",
        "9": "adrenal_gland_left",
        "10": "lung_upper_lobe_left",
        "11": "lung_lower_lobe_left",
        "12": "lung_upper_lobe_right",
        "13": "lung_middle_lobe_right",
        "14": "lung_lower_lobe_right",
        "15": "esophagus",
        "16": "trachea",
        "17": "thyroid_gland",
        "18": "small_bowel",
        "19": "duodenum",
        "20": "colon",
        "21": "urinary_bladder",
        "22": "prostate",
        "23": "kidney_cyst_left",
        "24": "kidney_cyst_right",
        "25": "sacrum",
        "26": "vertebrae_S1",
        "27": "vertebrae_L5",
        "28": "vertebrae_L4",
        "29": "vertebrae_L3",
        "30": "vertebrae_L2",
        "31": "vertebrae_L1",
        "32": "vertebrae_T12",
        "33": "vertebrae_T11",
        "34": "vertebrae_T10",
        "35": "vertebrae_T9",
        "36": "vertebrae_T8",
        "37": "vertebrae_T7",
        "38": "vertebrae_T6",
        "39": "vertebrae_T5",
        "40": "vertebrae_T4",
        "41": "vertebrae_T3",
        "42": "vertebrae_T2",
        "43": "vertebrae_T1",
        "44": "vertebrae_C7",
        "45": "vertebrae_C6",
        "46": "vertebrae_C5",
        "47": "vertebrae_C4",
        "48": "vertebrae_C3",
        "49": "vertebrae_C2",
        "50": "vertebrae_C1",
        "51": "heart",
        "52": "aorta",
        "53": "pulmonary_vein",
        "54": "brachiocephalic_trunk",
        "55": "subclavian_artery_right",
        "56": "subclavian_artery_left",
        "57": "common_carotid_artery_right",
        "58": "common_carotid_artery_left",
        "59": "brachiocephalic_vein_left",
        "60": "brachiocephalic_vein_right",
        "61": "atrial_appendage_left",
        "62": "superior_vena_cava",
        "63": "inferior_vena_cava",
        "64": "portal_vein_and_splenic_vein",
        "65": "iliac_artery_left",
        "66": "iliac_artery_right",
        "67": "iliac_vena_left",
        "68": "iliac_vena_right",
        "69": "humerus_left",
        "70": "humerus_right",
        "71": "scapula_left",
        "72": "scapula_right",
        "73": "clavicula_left",
        "74": "clavicula_right",
        "75": "femur_left",
        "76": "femur_right",
        "77": "hip_left",
        "78": "hip_right",
        "79": "spinal_cord",
        "80": "gluteus_maximus_left",
        "81": "gluteus_maximus_right",
        "82": "gluteus_medius_left",
        "83": "gluteus_medius_right",
        "84": "gluteus_minimus_left",
        "85": "gluteus_minimus_right",
        "86": "autochthon_left",
        "87": "autochthon_right",
        "88": "iliopsoas_left",
        "89": "iliopsoas_right",
        "90": "brain",
        "91": "skull",
        "92": "rib_left_1",
        "93": "rib_left_2",
        "94": "rib_left_3",
        "95": "rib_left_4",
        "96": "rib_left_5",
        "97": "rib_left_6",
        "98": "rib_left_7",
        "99": "rib_left_8",
        "100": "rib_left_9",
        "101": "rib_left_10",
        "102": "rib_left_11",
        "103": "rib_left_12",
        "104": "rib_right_1",
        "105": "rib_right_2",
        "106": "rib_right_3",
        "107": "rib_right_4",
        "108": "rib_right_5",
        "109": "rib_right_6",
        "110": "rib_right_7",
        "111": "rib_right_8",
        "112": "rib_right_9",
        "113": "rib_right_10",
        "114": "rib_right_11",
        "115": "rib_right_12",
        "116": "sternum",
        "117": "costal_cartilages",
        "200": "nodal_below_cib",
        "201": "nodal_above_cib"
    },
    "bone": {
        "25": "sacrum",
        "26": "vertebrae_S1",
        "27": "vertebrae_L5",
        "28": "vertebrae_L4",
        "29": "vertebrae_L3",
        "30": "vertebrae_L2",
        "31": "vertebrae_L1",
        "32": "vertebrae_T12",
        "33": "vertebrae_T11",
        "34": "vertebrae_T10",
        "35": "vertebrae_T9",
        "36": "vertebrae_T8",
        "37": "vertebrae_T7",
        "38": "vertebrae_T6",
        "39": "vertebrae_T5",
        "40": "vertebrae_T4",
        "41": "vertebrae_T3",
        "42": "vertebrae_T2",
        "43": "vertebrae_T1",
        "44": "vertebrae_C7",
        "45": "vertebrae_C6",
        "46": "vertebrae_C5",
        "47": "vertebrae_C4",
        "48": "vertebrae_C3",
        "49": "vertebrae_C2",
        "50": "vertebrae_C1",
        "69": "humerus_left",
        "70": "humerus_right",
        "71": "scapula_left",
        "72": "scapula_right",
        "73": "clavicula_left",
        "74": "clavicula_right",
        "75": "femur_left",
        "76": "femur_right",
        "77": "hip_left",
        "78": "hip_right",
        "91": "skull",
        "92": "rib_left_1",
        "93": "rib_left_2",
        "94": "rib_left_3",
        "95": "rib_left_4",
        "96": "rib_left_5",
        "97": "rib_left_6",
        "98": "rib_left_7",
        "99": "rib_left_8",
        "100": "rib_left_9",
        "101": "rib_left_10",
        "102": "rib_left_11",
        "103": "rib_left_12",
        "104": "rib_right_1",
        "105": "rib_right_2",
        "106": "rib_right_3",
        "107": "rib_right_4",
        "108": "rib_right_5",
        "109": "rib_right_6",
        "110": "rib_right_7",
        "111": "rib_right_8",
        "112": "rib_right_9",
        "113": "rib_right_10",
        "114": "rib_right_11",
        "115": "rib_right_12",
        "116": "sternum"
    },
    "prostate": {
        "22": "prostate"
    },
    "nodal_below_cib": {
        "200": "nodal_below_cib"
    },    
    "nodal_above_cib": {
        "201": "nodal_above_cib"
    },
    "visceral": {
        "1": "spleen",
        "2": "kidney_right",
        "3": "kidney_left",
        "4": "gallbladder",
        "5": "liver",
        "6": "stomach",
        "7": "pancreas",
        "8": "adrenal_gland_right",
        "9": "adrenal_gland_left",
        "10": "lung_upper_lobe_left",
        "11": "lung_lower_lobe_left",
        "12": "lung_upper_lobe_right",
        "13": "lung_middle_lobe_right",
        "14": "lung_lower_lobe_right",
        "15": "esophagus",
        "16": "trachea",
        "17": "thyroid_gland",
        "18": "small_bowel",
        "19": "duodenum",
        "20": "colon",
        "21": "urinary_bladder",
        "23": "kidney_cyst_left",
        "24": "kidney_cyst_right",
        "51": "heart",
        "52": "aorta",
        "53": "pulmonary_vein",
        "54": "brachiocephalic_trunk",
        "55": "subclavian_artery_right",
        "56": "subclavian_artery_left",
        "57": "common_carotid_artery_right",
        "58": "common_carotid_artery_left",
        "59": "brachiocephalic_vein_left",
        "60": "brachiocephalic_vein_right",
        "61": "atrial_appendage_left",
        "62": "superior_vena_cava",
        "63": "inferior_vena_cava",
        "64": "portal_vein_and_splenic_vein",
        "65": "iliac_artery_left",
        "66": "iliac_artery_right",
        "67": "iliac_vena_left",
        "68": "iliac_vena_right",
        "79": "spinal_cord",
        "80": "gluteus_maximus_left",
        "81": "gluteus_maximus_right",
        "82": "gluteus_medius_left",
        "83": "gluteus_medius_right",
        "84": "gluteus_minimus_left",
        "85": "gluteus_minimus_right",
        "86": "autochthon_left",
        "87": "autochthon_right",
        "88": "iliopsoas_left",
        "89": "iliopsoas_right",
        "90": "brain",
        "117": "costal_cartilages"
    }
}

## LESION EXPANSION ##

# Connected Component Analysis with organ-aware Fast Marching expansion from SUVmax voxel

def run_cca_fast_marching(lesion_mask, pet_image, organ_total, suv_threshold, local_thresh_ratio=0.5):
    """
    Identifies connected components and applies organ-aware Fast Marching expansion from SUVmax voxel.
    
    For each lesion component:
    1. Identify which organ it belongs to (most common organ label within the component)
    2. Only allow expansion within that same organ (or background if no organ)
    3. Apply SUV-based threshold (max of global threshold or local_thresh_ratio * SUVmax)
    """
    try:
        struct = generate_binary_structure(3, 1)
    except Exception:
        struct = generate_binary_structure(2, 1)

    labeled, ncomp = label(lesion_mask, structure=struct)
    result = np.copy(lesion_mask).astype(bool)

    if ncomp == 0:
        return result.astype(np.uint8)

    for comp_id in range(1, ncomp + 1):
        comp_mask = (labeled == comp_id)
        if not np.any(comp_mask):
            continue
            
        # Identify which organ this component is in
        if organ_total is not None:
            organ_labels_in_comp = organ_total[comp_mask]
            if organ_labels_in_comp.size > 0:
                # Find the most common organ label (excluding 0/background)
                unique_labels, counts = np.unique(organ_labels_in_comp, return_counts=True)
                non_zero_mask = unique_labels > 0
                if np.any(non_zero_mask):
                    primary_organ = unique_labels[non_zero_mask][np.argmax(counts[non_zero_mask])]
                else:
                    primary_organ = 0  # Component is in background
            else:
                primary_organ = 0
        else:
            primary_organ = 0
            
        # Create organ constraint mask: allow expansion only within same organ or background
        if organ_total is not None and primary_organ > 0:
            # Allow expansion within the same organ only
            organ_constraint = (organ_total == primary_organ)
        else:
            # If in background, forbid expansion into any organ
            if organ_total is not None:
                organ_constraint = (organ_total == 0)
            else:
                organ_constraint = np.ones_like(pet_image, dtype=bool)
        
        # Extract SUV values inside the component
        comp_suv = pet_image * comp_mask
        max_suv = float(np.max(comp_suv))
        if max_suv <= 0 or np.isnan(max_suv):
            continue

        # Choose conservative local threshold
        local_thresh = max(suv_threshold, max_suv * local_thresh_ratio)

        # Threshold the PET image with organ constraint
        thr_mask = np.logical_and(pet_image >= local_thresh, organ_constraint)
        
        # Label thresholded regions
        thr_labeled, thr_n = label(thr_mask, structure=struct)
        seed_idx = np.unravel_index(np.argmax(comp_suv), comp_suv.shape)
        seed_label = thr_labeled[seed_idx]
        
        if seed_label == 0:
            # Seed not included in thresholded mask; keep original component
            region = comp_mask
        else:
            # Take the connected region containing the seed
            region = thr_labeled == seed_label

        # Combine with result
        result = np.logical_or(result, region)

    return result.astype(np.uint8)

# Compute dynamic SUV threshold using liver uptake via GMM

def find_liver_reference_uptake(liver_mask, pet_image):
    """
    Computes dynamic SUV threshold using liver uptake via Gaussian Mixture Model.
    """
    liver_suvs = pet_image[liver_mask > 0].reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0).fit(liver_suvs)
    means = gmm.means_.flatten()
    dominant_mode = np.min(means)  # Conservative: assume lower mean is background
    return dominant_mode * 1.5  # Dynamic threshold factor

# Watershed filtering to segment contiguous high-uptake regions

def watershed_filtering(pet_image, percentile=99):
    """
    Applies watershed filtering to PET image to segment contiguous high-uptake regions.
    """
    # Simple, robust heuristic: threshold at a high percentile and label connected components.
    try:
        pct = np.nanpercentile(pet_image, percentile)
    except Exception:
        pct = np.nanpercentile(pet_image[np.isfinite(pet_image)], percentile)

    high_mask = pet_image >= max(pct, 0.0)
    try:
        struct = generate_binary_structure(3, 1)
        labeled, n = label(high_mask, structure=struct)
    except Exception:
        struct = generate_binary_structure(2, 1)
        labeled, n = label(high_mask, structure=struct)

    return labeled

# Organ-aware lesion expansion with triple constraints

def expand_lesion_segmentation(lesion_mask, 
                                pet_image, 
                                organ_total, 
                                suv_threshold, 
                                max_volume_factor=2.0,
                                watershed_labels=None):
    """
    Expands lesion mask with triple constraints to prevent over-expansion:
    
    1. **Organ-aware**: Each component can only expand within its primary organ (prevents crossing boundaries)
    2. **Volume-limited**: Expansion capped at max_volume_factor * original size (prevents full organ takeover)
    3. **Watershed-constrained**: Only expands within contiguous high-uptake regions (prevents leaking into distant areas)
    
    For each lesion component:
    - Identify which organ it belongs to (most common label within component)
    - Build forbidden mask: disallow expansion into different organs
    - Apply SUV threshold constraint
    - Apply watershed region constraint (if provided)
    - Iteratively dilate until volume limit or no valid candidates remain
    """
    from scipy.ndimage import binary_dilation
    
    if lesion_mask is None or lesion_mask.sum() == 0:
        return lesion_mask.astype(np.uint8)
    
    try:
        struct = generate_binary_structure(3, 1)
    except Exception:
        struct = generate_binary_structure(2, 1)
    
    lesion_labeled, ncomp = label(lesion_mask, structure=struct)
    final = np.zeros_like(lesion_mask, dtype=bool)
    
    for cid in range(1, ncomp + 1):
        comp = lesion_labeled == cid
        orig_vol = np.sum(comp)
        if orig_vol == 0:
            continue
        max_voxels = int(max_volume_factor * orig_vol)
        
        # Identify which organ this component is primarily in
        if organ_total is not None:
            organ_labels_in_comp = organ_total[comp]
            if organ_labels_in_comp.size > 0:
                unique_labels, counts = np.unique(organ_labels_in_comp, return_counts=True)
                non_zero_mask = unique_labels > 0
                if np.any(non_zero_mask):
                    primary_organ = unique_labels[non_zero_mask][np.argmax(counts[non_zero_mask])]
                else:
                    primary_organ = 0  # Component is in background
            else:
                primary_organ = 0
        else:
            primary_organ = 0
        
        # Build forbidden mask: forbid expansion into DIFFERENT organs
        if organ_total is None:
            forbidden = np.zeros_like(pet_image, dtype=bool)
        elif primary_organ > 0:
            # Lesion is in an organ: forbid expansion into different organs (but allow same organ + background)
            forbidden = np.logical_and(organ_total > 0, organ_total != primary_organ)
        else:
            # Lesion is in background: forbid expansion into ANY organ
            forbidden = (organ_total > 0)
        
        # Watershed constraint: only expand within connected high-uptake regions
        if watershed_labels is not None:
            overlapping_watershed_labels = np.unique(watershed_labels[comp])
            overlapping_watershed_labels = overlapping_watershed_labels[overlapping_watershed_labels > 0]
            
            if len(overlapping_watershed_labels) > 0:
                watershed_zone = np.isin(watershed_labels, overlapping_watershed_labels)
            else:
                watershed_zone = np.ones_like(lesion_mask, dtype=bool)
        else:
            watershed_zone = np.ones_like(lesion_mask, dtype=bool)
        
        # Iterative expansion with all constraints
        current = comp.copy()
        iterations = 0
        while True:
            iterations += 1
            
            # Candidate expansion: one-voxel dilation
            dilated = binary_dilation(current, structure=struct)
            candidate = np.logical_and(dilated, np.logical_not(current))
            
            # Apply triple constraints:
            # 1. SUV threshold
            candidate = np.logical_and(candidate, pet_image >= suv_threshold)
            # 2. Organ boundary (no crossing into different organs)
            candidate = np.logical_and(candidate, np.logical_not(forbidden))
            # 3. Watershed regions (stay within contiguous high-uptake areas)
            candidate = np.logical_and(candidate, watershed_zone)
            
            # Add candidates
            new = np.logical_or(current, candidate)
            
            # Volume limit check
            if np.sum(new) > max_voxels:
                break
            if np.array_equal(new, current):
                break
                
            current = new
            
            # Safety limit on iterations
            if iterations > 50:
                break
        
        final = np.logical_or(final, current)
    
    return final.astype(np.uint8)

# Main function to expand segmentations in a directory

def expand_segmentations(
    lesion_dir,
    organ_dir,
    ct_map,
    pet_map,
    *,
    max_volume_factor=3.5,
    local_thresh_ratio=0.8,
    suv_default=3.0,
    watershed_percentile=99.2,
    watershed=True,
    output_pred_dir_expanded=None,
    overwrite=False,
    verbose=False
):
    """
    Expands lesion segmentations using organ-aware constraints, dynamic SUV thresholding,
    watershed filtering, and fast marching expansion.
    
    **Expansion Strategy (Triple Constraints):**
    1. **CCA + Region Growing**: Expand from SUVmax voxel using adaptive threshold
    2. **Organ Boundaries**: Never cross into different organs (liver mets stay in liver, etc.)
    3. **Watershed Filtering**: Stay within contiguous high-uptake PET regions (optional)
    4. **Volume Limiting**: Cap expansion to prevent full organ takeover

    Parameters:
    - lesion_dir: Directory containing predicted lesion NIfTI files.
    - organ_dir: Directory containing TotalSegmentator organ NIfTI files.
    - ct_map: Dictionary mapping case names to CT file paths.
    - pet_map: Dictionary mapping case names to PET file paths.
    - max_volume_factor: Maximum growth factor per component (prevents full organ takeover).
    - local_thresh_ratio: Ratio of SUVmax used for local region growing (CCA step).
    - suv_default: Fallback SUV threshold when liver/aorta references are missing.
    - watershed_percentile: Percentile used for high-uptake coarse regions.
    - watershed: If True, apply watershed constraint to keep expansion within contiguous hotspots.
    - output_pred_dir_expanded: Output directory for expanded segmentations. If None, uses lesion_dir with '_expanded' suffix.
    - overwrite: If True, overwrite existing expanded segmentations.
    - verbose: If True, print progress messages.
    """

    if output_pred_dir_expanded is None:
        output_pred_dir_expanded = lesion_dir.rstrip(os.sep) + "_expanded"
    os.makedirs(output_pred_dir_expanded, exist_ok=True)

    for file_name in tqdm(os.listdir(lesion_dir), desc="Expanding segmentations"):
        if not file_name.endswith('.nii.gz'):
            continue

        case_name = file_name.split('.')[0]
        lesion_path = os.path.join(lesion_dir, file_name)

        out_name = f"{case_name}.nii.gz"
        out_path = os.path.join(output_pred_dir_expanded, out_name)
        if os.path.exists(out_path):
            if not overwrite:
                # Skip processing if output already exists
                print(f"Skipping already-expanded case: {case_name} at {out_path}")
                continue
            else:
                if verbose:
                    print(f"Overwriting already-expanded case: {case_name} at {out_path}")

        # Retrieve CT and PET paths from maps
        ct_path = ct_map.get(case_name)
        pet_path = pet_map.get(case_name)

        if not pet_path or not ct_path or not os.path.exists(pet_path) or not os.path.exists(ct_path):
            print(f"Missing PET/CT for case {case_name}")
            continue

        lesion_img = nib.load(lesion_path)
        lesion_mask = lesion_img.get_fdata()
        pet_image = nib.load(pet_path).get_fdata()

        # Load organ_total (TotalSegmentator) if available; prefer the single *_total.nii.gz file
        organ_total_path = os.path.join(organ_dir, f"{case_name}_total.nii.gz")
        organ_total_data = None
        if os.path.exists(organ_total_path):
            try:
                organ_total_nib = nib.load(organ_total_path)
                # Resample organ_total to lesion image if shapes differ
                if organ_total_nib.shape != lesion_img.shape:
                    organ_total_data = resample_input_to_target(organ_total_nib, lesion_img)
                else:
                    organ_total_data = organ_total_nib.get_fdata().astype(np.int32)
            except Exception:
                organ_total_data = None

        # If organ_total is not present, build a coarse organ_total from any individual organ files
        if organ_total_data is None:
            organ_total_data = np.zeros_like(pet_image, dtype=np.int32)
            for organ_file in os.listdir(organ_dir):
                if organ_file.startswith(case_name) and organ_file.endswith('.nii.gz'):
                    try:
                        organ_mask = nib.load(os.path.join(organ_dir, organ_file)).get_fdata()
                        organ_total_data[organ_mask > 0] = 1
                    except Exception:
                        # skip unreadable organ files
                        continue

        # Obtain liver mask using TotalSegmentator labelling convention (5 == liver)
        liver_mask = (organ_total_data == 5).astype(np.uint8)
        suv_threshold = None
        if liver_mask.sum() > 0:
            # Prefer the robust GMM-based approach when liver exists
            try:
                suv_threshold = find_liver_reference_uptake(liver_mask, pet_image)
            except Exception:
                suv_threshold = None

        # Fallback: if liver not found or GMM failed, try aorta (label 52) and use IQR mean
        if suv_threshold is None or liver_mask.sum() == 0:
            aorta_mask = (organ_total_data == 52).astype(np.uint8)
            if aorta_mask.sum() > 0:
                vals = pet_image[aorta_mask > 0]
                if vals.size > 0:
                    q1, q3 = np.percentile(vals, [25, 75])
                    iqr_vals = vals[(vals >= q1) & (vals <= q3)]
                    if iqr_vals.size > 0:
                        suv_threshold = float(np.nanmean(iqr_vals))
                    else:
                        suv_threshold = None
                else:
                    suv_threshold = None
            else:
                suv_threshold = None

        # If still no threshold, use conservative default
        if suv_threshold is None:
            suv_threshold = float(suv_default)
            print(f"[INFO] Liver and aorta masks missing or empty for {case_name}; using fallback SUV threshold = {suv_threshold}")

        # STEP 1: CCA + Fast Marching with organ awareness
        expanded_mask = run_cca_fast_marching(lesion_mask, pet_image, organ_total_data, suv_threshold, local_thresh_ratio=local_thresh_ratio)

        # STEP 2: Watershed filtering (optional) - identifies contiguous high-uptake regions
        if watershed:
            watershed_labels = watershed_filtering(pet_image, percentile=watershed_percentile)
        else:
            watershed_labels = None

        # STEP 3: Constrained iterative expansion with organ boundaries + volume limits + watershed
        final_mask = expand_lesion_segmentation(expanded_mask, 
                                                pet_image, 
                                                organ_total_data, 
                                                suv_threshold, 
                                                max_volume_factor=max_volume_factor,
                                                watershed_labels=watershed_labels)

        # Save final expanded segmentation into the chosen output directory
        try:
            final_img = nib.Nifti1Image(final_mask.astype(np.uint8), affine=lesion_img.affine, header=lesion_img.header)
            nib.save(final_img, out_path)
            # print(f"Processed and saved expanded segmentation for case: {case_name} -> {out_name}")
        except Exception as e:
            print(f"Warning: Failed to save expanded segmentation for {case_name}: {e}")
    return output_pred_dir_expanded


## Apply SUV threshold to segmentation outputs ##
def apply_suv_threshold(pet_map, 
                        output_pred_dir, 
                        suv_thresh, 
                        verbose, 
                        overwrite):
    seg_files = [f for f in os.listdir(output_pred_dir) if f.endswith('.nii.gz')]

    for seg_file in seg_files:
        seg_path = os.path.join(output_pred_dir, seg_file)
        if verbose:
            print(f"Processing segmentation file: {shorten_path(seg_path)}")
        seg_img = nib.load(seg_path)
        seg_data = seg_img.get_fdata()

        seg_base = os.path.splitext(os.path.splitext(seg_file)[0])[0]
        pt_path = pet_map.get(seg_base)
        if not pt_path or not os.path.exists(pt_path):
            print(f"Warning: Corresponding PET image not found for {shorten_path(seg_file)} in {pt_path}. Skipping.")
            continue
        pt_img = nib.load(pt_path)
        pt_data = pt_img.get_fdata()

        # Apply SUV threshold
        mask = seg_data > 0
        suv_mask = pt_data >= suv_thresh
        new_mask = np.zeros_like(seg_data, dtype=np.uint8)
        new_mask[mask & suv_mask] = 1

        new_seg_img = nib.Nifti1Image(new_mask, affine=seg_img.affine, header=seg_img.header)

        if not overwrite:
            backup_dir = os.path.join(output_pred_dir, "backups_no_threshold")
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, seg_file)
            if os.path.exists(backup_path):
                print(f"Backup already exists at {shorten_path(backup_path)}. Skipping backup.")
            else:
                print(f"Moving original segmentation to backup folder: {shorten_path(backup_path)}")
                os.rename(seg_path, backup_path)  # move original to backup folder

        nib.save(new_seg_img, seg_path)

        if verbose:
            print(f"Applied SUV threshold {suv_thresh} to {seg_file}")
            # Print how many voxels were removed (absolute and percentage)
            num_removed = np.sum(mask) - np.sum(new_mask)
            percent_removed = (num_removed / np.sum(mask)) * 100
            print(f"  Removed {num_removed} voxels ({percent_removed:.2f}%) from segmentation.")

def seg_to_rtstruct(output_pred_dir, 
                    ct_dicom_case_map,
                    ct_map, 
                    rtstruct_dir, 
                    verbose,
                    overwrite):
    seg_files = [f for f in os.listdir(output_pred_dir) if f.endswith('.nii.gz')]

    # Collect skipped cases and their messages so we can write a CSV report at the end
    skipped_cases = []  # list of (case_name, message)

    for seg_file in seg_files:
        seg_nifti_path = Path(output_pred_dir) / seg_file
        # Extract case_name from seg_file (remove .nii.gz and any suffixes)
        case_name = seg_file.split('.nii')[0]
        # If ct_map is provided, use its keys for canonical case names
        if ct_map and case_name not in ct_map:
            # Try to match by prefix
            for k in ct_map.keys():
                if seg_file.startswith(k):
                    case_name = k
                    break
        # Get CT DICOM path from ct_dicom_case_map
        ct_dicom_dir = ct_dicom_case_map.get(case_name)
        # print(f"CT DICOM dir for case {case_name}: {ct_dicom_dir}")
        if not ct_dicom_dir:
            msg = f"No CT DICOM path found for case {case_name} at {ct_dicom_dir}. Skipping RTSTRUCT conversion for {seg_file}."
            print(f"Warning: {msg}")
            skipped_cases.append((case_name, msg))
            continue
        # Read first CT DICOM file to extract patient info
        ct_dicom_files = [f for f in os.listdir(ct_dicom_dir)]
        patient_id = case_name
        patient_name = case_name
        series_description = f'PSMASegmentator_RTSTRUCT_{case_name}'
        if ct_dicom_files:
            first_ct_dicom = os.path.join(ct_dicom_dir, ct_dicom_files[0])
            try:
                ds_ct = pydicom.dcmread(first_ct_dicom, stop_before_pixels=True)
                patient_id = getattr(ds_ct, 'PatientID', case_name)
                patient_name = getattr(ds_ct, 'PatientName', case_name)
                series_description = getattr(ds_ct, 'SeriesDescription', f'PSMASegmentator_RTSTRUCT_{case_name}')
            except Exception as e:
                print(f"Warning: Could not read CT DICOM file {first_ct_dicom}: {e}")
        else:
            print(f"Warning: No CT DICOM files found in {ct_dicom_dir}. Using default patient info.")
        rtstruct_name = f"{case_name}"
        rtstruct_dir_case = os.path.join(rtstruct_dir, rtstruct_name)
        # Check if RTSTRUCT already exists
        if os.path.exists(rtstruct_dir_case) and not overwrite:
            # If path is a directory, look for .dcm file inside
            if os.path.isdir(rtstruct_dir_case):
                dcm_files = [f for f in os.listdir(rtstruct_dir_case) if f.endswith('.dcm')]
                if dcm_files:
                    dcm_file_path = os.path.join(rtstruct_dir_case, dcm_files[0])
                    try:
                        ds = pydicom.dcmread(dcm_file_path, stop_before_pixels=True)
                        if getattr(ds, 'Modality', None) == 'RTSTRUCT':
                            msg = f"RTSTRUCT already exists for case {case_name} at {shorten_path(dcm_file_path)}. Skipping conversion."
                            print(msg)
                            # skipped_cases.append((case_name, msg))
                            continue
                    except Exception as e:
                        print(f"Warning: Could not read DICOM file {dcm_file_path}: {e}")
                else:
                    if verbose:
                        print(f"No DICOM files found in {rtstruct_dir_case}. Proceeding with conversion.")
            else:
                msg = f"Invalid RTSTRUCT path {rtstruct_dir_case}. Skipping conversion."
                print(msg)
                skipped_cases.append((case_name, msg))
                continue
        else:
            os.makedirs(rtstruct_dir_case, exist_ok=True)

        try:
            command = [
                'plastimatch', 'convert',
                '--input-ss-img', str(seg_nifti_path),
                '--referenced-ct', ct_dicom_dir,
                '--output-dicom', rtstruct_dir_case,
                '--series-description', str(series_description),
                '--patient-id', str(patient_id),
                '--patient-name', str(patient_name)
            ]
            subprocess.run(command, check=True)
            # Post-process RTSTRUCT to set structure name
            # Find the .dcm file in rtstruct_dir_case
            dcm_files = [f for f in os.listdir(rtstruct_dir_case) if f.endswith('.dcm')]
            if dcm_files:
                dcm_file_path = os.path.join(rtstruct_dir_case, dcm_files[0])
                try:
                    ds_rt = pydicom.dcmread(dcm_file_path)
                    # Set ROIName in StructureSetROISequence
                    if hasattr(ds_rt, 'StructureSetROISequence'):
                        for roi in ds_rt.StructureSetROISequence:
                            roi.ROIName = 'Total Tumor Burden'
                    # Set ROIName in RTROIObservationsSequence (optional)
                    if hasattr(ds_rt, 'RTROIObservationsSequence'):
                        for obs in ds_rt.RTROIObservationsSequence:
                            obs.ROIObservationLabel = 'Total Tumor Burden'
                    ds_rt.save_as(dcm_file_path)
                    if verbose:
                        print(f"Renamed structure in RTSTRUCT to 'Total Tumor Burden' for {case_name}.")
                except Exception as e:
                    print(f"Warning: Could not update structure name in RTSTRUCT {dcm_file_path}: {e}")
            if verbose:
                print(f"Converted {seg_nifti_path} to RTSTRUCT at {rtstruct_dir_case} using CT DICOMs from {ct_dicom_dir}")
        except subprocess.CalledProcessError as e:
            msg = f"ERROR: Plastimatch NIfTI→RTSTRUCT conversion failed for {seg_file}: {e}"
            print(msg)
            skipped_cases.append((case_name, msg))
            continue

    # After processing all segmentations, write a CSV with skipped cases (if any)
    try:
        if skipped_cases:
            os.makedirs(rtstruct_dir, exist_ok=True)
            import csv as _csv
            csv_path = os.path.join(rtstruct_dir, "rtstruct_skipped_cases.csv")
            with open(csv_path, "w", newline='') as _f:
                _writer = _csv.writer(_f)
                _writer.writerow(["case_name", "message"])
                for case_name, msg in skipped_cases:
                    _writer.writerow([case_name, msg])
            if verbose:
                print(f"Saved RTSTRUCT skip report to {csv_path}")
    except Exception as e:
        # Non-fatal: report but don't raise
        print(f"Warning: Could not write RTSTRUCT skip CSV to {rtstruct_dir}: {e}")

## Generate organ segmentations using TotalSegmentator ##
def generate_organ_segmentations(ct_map, organ_dir, 
                                    device, fast, verbose):
    if not device == 'cpu':
        device = 'gpu'

    organ_dir_path = Path(organ_dir)
    organ_dir_path.mkdir(parents=True, exist_ok=True)

    # Use ct_map for robust case naming
    # Best practice: save organ segmentations in a dedicated organ_dir, not alongside CT, for clarity and separation of outputs
    total_time_all_cases = 0.0
    n_cases_run = 0

    for case_base, ct_path in ct_map.items():
        ct_path_obj = Path(ct_path).expanduser()
        out_path_total = organ_dir_path / f"{case_base}_total.nii.gz"
        if out_path_total.exists():
            print(f"Organ segmentation already exists for {shorten_path(ct_path)} at {shorten_path(out_path_total)}. Skipping.")
            continue
        if verbose:
            print(f"Saving organ segmentation for {shorten_path(ct_path)} to {shorten_path(out_path_total)}")
        matching_segs = [p for p in organ_dir_path.glob(f"{case_base}*.nii.gz") if p.is_file()]
        valid_total_found = False
        for seg_file in matching_segs:
            seg_base = seg_file.name[:-7] if seg_file.name.endswith('.nii.gz') else seg_file.stem
            if '_total' not in seg_base:
                seg_path = seg_file
                try:
                    seg_img = nib.load(seg_path)
                    seg_data = seg_img.get_fdata()
                    unique_labels = np.unique(seg_data)
                    if (len(unique_labels) == 117) or (unique_labels.max() == 117):
                        new_path = organ_dir_path / f"{case_base}_total.nii.gz"
                        seg_path.replace(new_path)
                        print(f"Found valid TotalSegmentator output for {case_base} without _total suffix. Renamed to: {shorten_path(new_path)}")
                        valid_total_found = True
                        break
                except Exception as e:
                    print(f"Could not check file {seg_path}: {e}")
        if valid_total_found:
            continue

        command = [
            "TotalSegmentator",
            "-i", os.fspath(ct_path_obj),
            "-o", os.fspath(out_path_total),
            "--ta", "total",
            "--ml",
            "-d", device
        ]
        if fast:
            command.append("--fast")
            print("Running TotalSegmentator in 'fast' mode.")
        # If running on Windows, reduce saving threads to 1 and use force_split:
        if os.name == 'nt':
            command.append("--nr_thr_saving")
            command.append("1")
            command.append("--force_split")
            print("Running TotalSegmentator with `--nr_thr_saving 1` and `--force_split` (not recommended for small images) for Windows compatibility.")

        # Measure wall-clock time for TotalSegmentator invocation (only)
        t0 = time.perf_counter()
        run_kwargs = {"check": False, "text": True}
        if not verbose:
            run_kwargs["capture_output"] = True
        proc = subprocess.run(command, **run_kwargs)
        ret = proc.returncode
        t1 = time.perf_counter()
        elapsed = t1 - t0
        total_time_all_cases += elapsed
        n_cases_run += 1

        if ret != 0:
            print(f"Warning: TotalSegmentator returned non-zero exit code {ret} for case {case_base}")
            if not verbose and getattr(proc, "stderr", None):
                err = proc.stderr.strip()
                if err:
                    print(f"TotalSegmentator stderr for {case_base}: {err}")
        print(f"TotalSegmentator time for {case_base}: {elapsed:.3f} s")

    # Summary timing for all TotalSegmentator runs
    if n_cases_run > 0:
        print(f"Total TotalSegmentator time: {total_time_all_cases:.3f} s for {n_cases_run} cases (avg {total_time_all_cases/n_cases_run:.3f} s/case)")

## Helper functions ##
def round_sig(x, sig=6):
    if x == 0:
        return 0.0
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)

def get_nifti_fdata(path, verbose):
    img = nib.load(path)
    data = img.get_fdata()
    if verbose:
        logging.info(f"Loaded NIFTI file from {path}")
        header = img.header
        logging.info(f"  Shape         : {img.shape} (Z, Y, X)")
        logging.info(f"  Voxel spacing : {header.get_zooms()} (Z, Y, X) mm")
        logging.info(f"  Data type     : {header.get_data_dtype()}")
        logging.info(f"  Min/Max value : {np.min(data)} / {np.max(data)}")
    return data

def crop_with_margin(lesion_mask, lesion_seg, 
                        organ_total,
                        margin):
    lesion_coords = np.argwhere(lesion_mask)
    minz, miny, minx = np.maximum(lesion_coords.min(axis=0) - margin, 0)
    maxz, maxy, maxx = np.minimum(lesion_coords.max(axis=0) + margin + 1, lesion_seg.shape)
    
    lesion_mask_crop = lesion_mask[minz:maxz, miny:maxy, minx:maxx]
    organ_total_crop = organ_total[minz:maxz, miny:maxy, minx:maxx]

    return lesion_mask_crop, organ_total_crop

def compute_bifurcation_z(organ_seg_total, artery_labels=(65, 66)):
    bifurcation_mask = np.isin(organ_seg_total, artery_labels)
    coords = np.argwhere(bifurcation_mask)

    if coords.size == 0:
        logging.warning("No iliac vessels found — cannot compute bifurcation.")
        return None

    z_left = coords[organ_seg_total[tuple(coords.T)] == artery_labels[0], 2]
    z_right = coords[organ_seg_total[tuple(coords.T)] == artery_labels[1], 2]

    if z_left.size == 0 or z_right.size == 0:
        logging.warning("One or both iliac arteries not found.")
        return None

    z_slices = np.unique(coords[:, 2])
    common_z_slices = []

    for z in z_slices:
        slice_mask = organ_seg_total[:, :, z]
        has_left = np.any(slice_mask == artery_labels[0])
        has_right = np.any(slice_mask == artery_labels[1])
        if has_left and has_right:
            common_z_slices.append(z)

    if not common_z_slices:
        logging.warning("No common z-slice where both iliac arteries are present.")
        return None

    bifurcation_z = max(common_z_slices)  # most inferior common slice

    return bifurcation_z

def calc_suv_metrics(pt_img, mask_array):
    if np.sum(mask_array) == 0:
        return {}

    # Get PET voxel data and spacing
    pet_array = pt_img.get_fdata()
    spacing = pt_img.header.get_zooms()  # (x, y, z) spacing

    # Make sure mask and PET shapes align
    assert pet_array.shape == mask_array.shape, f"Shape mismatch: PET {pet_array.shape}, mask {mask_array.shape}"

    mask_array = mask_array.astype(bool)

    voxel_volume_mm3 = np.prod(spacing)
    voxel_volume_ml = voxel_volume_mm3 / 1000

    suv_values = pet_array[mask_array] # this a macro-level SUV extraction
    suv_mean = np.mean(suv_values)
    suv_max = np.max(suv_values)
    ttv = np.sum(mask_array) * voxel_volume_ml  # Total Tumor Volume in mL
    tlu = ttv * suv_mean  # Total Lesion Uptake
    # Calculate TLQ, given by dividing TTV by SUVmean
    tlq = ttv / suv_mean if suv_mean != 0 else 0

    return {
        'SUVmean': round_sig(suv_mean, 6),
        'SUVmax': round_sig(suv_max, 6),
        'TLU': round_sig(tlu, 6),
        'TLQ': round_sig(tlq, 6)
    }

## Lesion classification and metric calculation ##
def calc_lesion_volume(pt_img, mask_img):
    """
    Calculates volume of a binary lesion mask in cubic centimeters (cm³).
    """
    mask_array = mask_img
    if np.sum(mask_array) == 0:
        return 0.0

    spacing = pt_img.header.get_zooms()  # (x, y, z) spacing in mm
    voxel_volume_mm3 = np.prod(spacing)
    lesion_voxel_count = np.sum(mask_array)
    lesion_volume_cm3 = (lesion_voxel_count * voxel_volume_mm3) / 1000.0

    return lesion_volume_cm3

def classify_lesion(
        pred_label, # lesion number
        pred_lesion_crop, # cropped lesion mask
        pred_lesion, # full lesion mask
        organ_total_crop, # cropped organ mask
        organ_total, # full organ mask
        label_dict_total, # organ label dictionary
        verbose,
        overlap_threshold=0.6,
    ):
    """
    Classifies a lesion based on overlap with organs or relative position to CIB (if nodal).
    """
    # Count overlaps
    overlap_counts = defaultdict(int)
    for organ_label, organ_name in label_dict_total.items():
        overlap_counts[f"{organ_label}: {organ_name}"] += np.sum(
            (organ_total_crop == organ_label) & pred_lesion_crop
        )

    lesion_voxel_count = np.sum(pred_lesion_crop)

    max_overlap_label = max(overlap_counts, key=overlap_counts.get, default=None)
    max_overlap_voxels = overlap_counts[max_overlap_label] if max_overlap_label else 0
    max_overlap_ratio = max_overlap_voxels / lesion_voxel_count

    if verbose:
        logging.info(f"\tLesion {pred_label}: Max overlap = {max_overlap_voxels} voxels "
                        f"({max_overlap_ratio:.2%} of lesion volume)")

    # If maximum overlap ratio is below the threshold → use nodal logic
    if max_overlap_voxels == 0 or max_overlap_ratio < overlap_threshold:
        if verbose:
            logging.info(f"\tLesion {pred_label}: Overlap below threshold ({overlap_threshold:.2%}), using nodal logic...")

        cib_z = compute_bifurcation_z(organ_total)
        if cib_z is not None:
            lesion_coords = np.argwhere(pred_lesion)
            lesion_z = np.mean(lesion_coords[:, 2])
            chosen_class = "nodal_above_cib" if lesion_z > cib_z else "nodal_below_cib"
            if verbose:
                logging.info(f"\t\tCIB z = {cib_z:.2f}, lesion z = {lesion_z:.2f} ({chosen_class})")
        else:
            # Don't raise here; instead warn and assign a placeholder so processing can continue
            logging.warning(
                "CIB z-slice not found for case - assuming iliac artery segmentations not provided. "
                "Cannot determine above/below for nodal lesion; assigning 'nodal_unknown_cib'. Review case files."
            )
            chosen_class = "nodal_unknown_cib"
            if verbose:
                logging.info(f"\t\tAssigned placeholder class '{chosen_class}' for lesion {pred_label} due to missing CIB")
    else:
        chosen_class = max_overlap_label
        if verbose:
            logging.info(f"\tLesion {pred_label}: Overlap above threshold ({overlap_threshold:.2%}), assigned to {chosen_class}")

    return chosen_class

def classify_case(pred_seg, 
                    organ_total, 
                    label_dict_total,
                    verbose,
                    overlap_threshold=0.6,
                    pt_img=None,
                    totalseg_to_bioregions=None,
                    margin_voxels=5
                    ):
    # Fix label keys if they come from JSON
    label_dict_total = {int(k): v for k, v in label_dict_total.items()}

    s3d = generate_binary_structure(3, 3)
    t0 = time.time()

    case_dict = {
                "lesions": {}, 
                "lesion_metrics": {
                    "site": {}, # each TS site
                    "region": {}, # regions in biomarker_regions
                    "patient": {} # SUV metrics
                    }
                }

    labeled_pred_seg, num_pred_lesions = label(pred_seg, structure=s3d)
    if verbose:
        logging.info(f"Connected components found in pred: {num_pred_lesions} (Labeling time: {time.time() - t0:.2f}s)")

    # Invert the mapping: site name -> region name(s)
    site_to_regions = defaultdict(list)
    for region, mapping in totalseg_to_bioregions.items():
        for ts_code_str, site_name in mapping.items():
            site_to_regions[site_name].append(region)

    for pred_label in range(1, num_pred_lesions + 1):
        pred_lesion = labeled_pred_seg == pred_label # lesion mask

        pred_lesion_crop, organ_total_crop = crop_with_margin(pred_lesion, pred_seg, 
                                                                organ_total,
                                                                margin_voxels)

        chosen_class = classify_lesion(
                            pred_label, pred_lesion_crop, pred_lesion,
                            organ_total_crop,
                            organ_total,
                            label_dict_total,
                            verbose=verbose,
                            overlap_threshold=overlap_threshold
                        )

        # Decompose class string into code and name if possible
        if ":" in chosen_class:
            ts_code, ts_name = chosen_class.split(": ", 1)
        elif chosen_class == 'nodal_below_cib':
            ts_code = "200"
            ts_name = chosen_class
        elif chosen_class == 'nodal_above_cib':
            ts_code = "201"
            ts_name = chosen_class
        else:
            ts_code = ""
            ts_name = chosen_class

        pred_vol = calc_lesion_volume(pt_img, pred_lesion)

        case_dict["lesions"][f"lesion_{pred_label}"] = {
            # "ts_code": ts_code.strip(), # not necessary info
            "ts_name": ts_name.strip(),
            "volume_cm3": pred_vol
        }

        # Update count for this ts_name
        if ts_name not in ("nodal_below_cib", "nodal_above_cib"):
            site_entry = case_dict["lesion_metrics"]["site"].get(ts_name.strip(), {})
            if not isinstance(site_entry, dict):
                site_entry = {}
            site_entry["lesion_count"] = site_entry.get("lesion_count", 0) + 1
            case_dict["lesion_metrics"]["site"][ts_name.strip()] = site_entry

        
        # Update region-level lesion counts using mapping from site → regions
        for region in site_to_regions.get(ts_name, []):
            region_entry = case_dict["lesion_metrics"]["region"].get(region, {})
            if not isinstance(region_entry, dict):
                region_entry = {}
            region_entry["lesion_count"] = region_entry.get("lesion_count", 0) + 1
            case_dict["lesion_metrics"]["region"][region] = region_entry

        if verbose:
            logging.info(f"\t\tClassified predicted lesion as {ts_name.strip()}")

    # Aggregate burden per site and region directly from lesions
    for lesion_data in case_dict["lesions"].values():
        ts_name = lesion_data["ts_name"]
        burden = lesion_data.get("volume_cm3", 0)

        # Add to site only if not CIB-specific
        if ts_name not in ("nodal_below_cib", "nodal_above_cib"):
            if ts_name not in case_dict["lesion_metrics"]["site"]:
                case_dict["lesion_metrics"]["site"][ts_name] = {}
            case_dict["lesion_metrics"]["site"][ts_name]["total_burden"] = (
                case_dict["lesion_metrics"]["site"][ts_name].get("total_burden", 0) + burden
            )

        # Add to each associated region
        for region in site_to_regions.get(ts_name, []):
            if region not in case_dict["lesion_metrics"]["region"]:
                case_dict["lesion_metrics"]["region"][region] = {}
            case_dict["lesion_metrics"]["region"][region]["total_burden"] = (
                case_dict["lesion_metrics"]["region"][region].get("total_burden", 0) + burden
            )

    # Calculate SUV metrics for each biomarker region
    if pt_img is not None:
        suv_metrics = calc_suv_metrics(pt_img, pred_seg)
        if suv_metrics:
            case_dict["lesion_metrics"]["patient"] = {
                "SUVmean": suv_metrics.get("SUVmean", 0),
                "SUVmax": suv_metrics.get("SUVmax", 0),
                "TLU": suv_metrics.get("TLU", 0),
                "TLQ": suv_metrics.get("TLQ", 0)
            }

    return case_dict

def lesion_classifier(lesion_dir, # from output_pred_dir
                        organ_dir, # newly created organ_dir
                        pet_map,
                        verbose,
                        case_filter=None,
                        overlap_threshold=0.6,
                        organ_suffix="_total"):
    
    label_dict_total = totalseg_labels_and_regions["whole_body"]

    lesion_cases = [f for f in os.listdir(lesion_dir) if f.endswith(".nii.gz")]
    logging.info(f"Found {len(lesion_cases)} lesion segmentations to process.")

    results = {}
    all_lesion_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for lesion_file in tqdm(lesion_cases, desc="Processing cases"):
        case_name = lesion_file.replace('.nii.gz', '')
        if case_filter and case_filter in case_name:
            logging.info(f"Skipping case {case_name} due to {case_filter} filter.")
            continue
        logging.info(f"\nProcessing case: {case_name}")

        lesion_path = os.path.join(lesion_dir, lesion_file)
        organ_total_path = os.path.join(organ_dir, f"{case_name}{organ_suffix}.nii.gz")

        pt_img = None
        pt_path = pet_map.get(case_name)
        if pt_path and os.path.exists(pt_path):
            pt_img = nib.load(pt_path)
        else:
            logging.warning(f"PET image not found for case {case_name} at {pt_path}. SUV metrics will be skipped.")

        if not os.path.exists(organ_total_path):
            logging.warning(f"Skipping case {case_name}: organ segmentation not found at {organ_total_path} (review output above to see if TotalSegmentator failed for this case).")
            continue

        lesion_seg = get_nifti_fdata(lesion_path, verbose=False)
        organ_total = get_nifti_fdata(organ_total_path, verbose=False)

        
        case_dict = classify_case(lesion_seg, 
                                    organ_total,
                                    label_dict_total,
                                    verbose=verbose,
                                    overlap_threshold=overlap_threshold,
                                    pt_img=pt_img,
                                    totalseg_to_bioregions=totalseg_labels_and_regions)

        # Aggregate site and region metrics
        for level in ["site", "region"]:
            for key, val in case_dict["lesion_metrics"].get(level, {}).items():
                all_lesion_metrics[level][key]["lesion_count"] += val.get("lesion_count", 0)
                all_lesion_metrics[level][key]["total_burden"] += val.get("total_burden", 0)

        results[case_name] = case_dict

    # Format total_burden values to 6 significant figures as strings
    for level in ["site", "region"]:
        for key in all_lesion_metrics[level]:
            if "total_burden" in all_lesion_metrics[level][key]:
                val = all_lesion_metrics[level][key]["total_burden"]
                all_lesion_metrics[level][key]["total_burden"] = round_sig(val, 6)

    results['All'] = {'lesion_metrics': all_lesion_metrics}

    return results


## LIVER CLASSIFICATION ##

## Helper functions ##

def load_nifti(path, 
                as_closest_canonical=True,
                return_nib=False):
    img = nib.load(path)
    if as_closest_canonical:
        img = nib.as_closest_canonical(img)
    return img if return_nib else img.get_fdata().astype(np.float32)

def pad(start, end, max_len, margin):
    return max(0, start - margin), min(max_len, end + margin)

def get_liver_bbox(liver_mask, margin=0):
    nonzero = np.nonzero(liver_mask)
    minz, maxz = np.min(nonzero[0]), np.max(nonzero[0]) + 1
    miny, maxy = np.min(nonzero[1]), np.max(nonzero[1]) + 1
    minx, maxx = np.min(nonzero[2]), np.max(nonzero[2]) + 1

    if margin > 0:
        minz, maxz = pad(minz, maxz, liver_mask.shape[0], margin)
        miny, maxy = pad(miny, maxy, liver_mask.shape[1], margin)
        minx, maxx = pad(minx, maxx, liver_mask.shape[2], margin)

    return slice(minz, maxz), slice(miny, maxy), slice(minx, maxx)

def resample_input_to_target(input_img, target_img):
    resampled_img = resample_from_to(input_img, target_img, order=0)  # nearest neighbor interpolation
    return resampled_img.get_fdata().astype(np.float32)

def ensure_spacing_and_orientation(img, target_spacing=TARGET_SPACING, target_orient=TARGET_ORIENT):
    """
    Resample and reorient NIfTI image if spacing or orientation don't match the target.
    Returns a NIfTI image with fixed spacing and orientation.
    """
    # Resample to target spacing if needed
    spacing = img.header.get_zooms()
    data = img.get_fdata()
    affine = img.affine

    if not np.allclose(spacing, target_spacing, atol=1e-3):
        scale = np.array(spacing) / np.array(target_spacing)
        new_shape = np.round(np.array(data.shape) * scale).astype(int)
        data = zoom(data, scale, order=1)
        affine = np.copy(affine)
        np.fill_diagonal(affine, list(target_spacing) + [1])
        img = nib.Nifti1Image(data, affine)

    # Reorient to target orientation if needed
    cur_orient = aff2axcodes(img.affine)
    if cur_orient != target_orient:
        trans = ornt_transform(axcodes2ornt(cur_orient), axcodes2ornt(target_orient))
        data = apply_orientation(img.get_fdata(), trans)
        img = nib.Nifti1Image(data, img.affine)

    return img

def pad_or_crop_to_match(batch, tolerance_ratio=1.75, verbose=False):
    imgs, labels = zip(*batch)
    
    shapes = [img.shape[1:] for img in imgs]  # (C, D, H, W) → (D, H, W)
    shape_areas = [np.prod(s) for s in shapes]

    if verbose:
        print(f"\nNew batch with {len(batch)} samples")
        for i, s in enumerate(shapes):
            print(f"  Sample {i}: shape = {s}, volume = {np.prod(s)}")

    # Default: pad to largest shape
    sorted_indices = sorted(range(len(shape_areas)), key=lambda i: shape_areas[i])
    largest_shape = shapes[sorted_indices[-1]]

    if len(shapes) == 1:
        target_shape = largest_shape
    else:
        second_largest_shape = shapes[sorted_indices[-2]]
        largest_vol = np.prod(largest_shape)
        second_largest_vol = np.prod(second_largest_shape)

        if largest_vol <= second_largest_vol * tolerance_ratio:
            target_shape = largest_shape
        else:
            target_shape = second_largest_shape
            if verbose:
                print("Largest is outlier, using second-largest as target.")

    target_d, target_h, target_w = target_shape

    processed_imgs = []
    for i, img in enumerate(imgs):
        c, d, h, w = img.shape
        crop_info = ""

        # Crop if this sample is too big
        if (d > target_d * tolerance_ratio or
            h > target_h * tolerance_ratio or
            w > target_w * tolerance_ratio):
            start_d = (d - target_d) // 2
            start_h = (h - target_h) // 2
            start_w = (w - target_w) // 2
            img = img[:, start_d:start_d+target_d,
                            start_h:start_h+target_h,
                            start_w:start_w+target_w]
            crop_info = " (cropped)"
            d, h, w = img.shape[1:]

        # Padding to match
        pad_d = target_d - d
        pad_h = target_h - h
        pad_w = target_w - w

        padding = [
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2,
            pad_d // 2, pad_d - pad_d // 2,
        ]
        img = F.pad(img, padding, mode='constant', value=0)
        processed_imgs.append(img)

        if verbose:
            print(f"  → Sample {i} final shape: {img.shape}{crop_info}")

    batch_imgs = torch.stack(processed_imgs)
    batch_labels = torch.stack(labels)
    return batch_imgs, batch_labels


def classify_liver_disease(ct_map,
                            pet_map,
                            organ_dir,
                            model_path,
                            device,
                            margin=5,
                            target_crop=None,
                            verbose=False
):
    # Load model
    model = DenseNet121(spatial_dims=3, in_channels=2, out_channels=1).to(device)
    if torch.cuda.is_available():
        checkpoint = torch.load(
            model_path,
            map_location=lambda storage, loc: storage.cuda(0),
            weights_only=False
        )
    else:
        checkpoint = torch.load(
            model_path, 
            map_location="cpu",
            weights_only=False
        )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    threshold = checkpoint.get("best_thresh", 0.5)
    transform = Compose([NormalizeIntensity(channel_wise=True)])

    if verbose:
        print(f"Loaded model from {model_path} with threshold {threshold:.4f}")
        print(f"Using device: {device}")

    # Iterate over ct_map for cases
    liver_classifications = {}
    for case_name in tqdm(sorted(ct_map.keys()), desc="Classifying liver disease"):
        ct_path = ct_map[case_name]
        pt_path = pet_map.get(case_name)
        seg_path = os.path.join(organ_dir, f"{case_name}_total.nii.gz")

        # Check file existence
        if not pt_path or not os.path.exists(pt_path):
            if verbose:
                print(f"Skipping {case_name}: Missing PET.")
            continue
        if not os.path.exists(seg_path):
            if verbose:
                print(f"Skipping {case_name}: Missing organ segmentation.")
            continue

        # Load and fix orientation/spacing
        ct_img = ensure_spacing_and_orientation(nib.load(ct_path))
        pt_img = ensure_spacing_and_orientation(nib.load(pt_path))
        seg_img = ensure_spacing_and_orientation(nib.load(seg_path))

        # Resample mask to CT space if needed
        seg_data = resample_input_to_target(seg_img, ct_img) if seg_img.shape != ct_img.shape else seg_img.get_fdata().astype(np.float32)

        # Extract liver mask
        liver_mask = (seg_data == 5.0).astype(np.uint8)
        if liver_mask.sum() == 0:
            if verbose:
                print(f"Skipping {case_name}: No liver voxels found.")
            continue

        # Resample PET to CT space if needed
        if pt_img.shape != ct_img.shape:
            pt_data = resample_input_to_target(ct_img, pt_img)
            if verbose:
                print(f"Resampled CT to match PET shape for {case_name}.")
        else:
            pt_data = pt_img.get_fdata().astype(np.float32)

        # Crop to liver bounding box
        z_s, y_s, x_s = get_liver_bbox(liver_mask, margin=margin)
        ct_crop = ct_img.get_fdata()[z_s, y_s, x_s].astype(np.float32)
        pt_crop = pt_data[z_s, y_s, x_s].astype(np.float32)

        # Stack into channels (CT, PET)
        img = np.stack([ct_crop, pt_crop], axis=0)  # shape: (2, D, H, W)

        # Pad or crop to fixed size if requested
        img_tensor = torch.tensor(img)
        if target_crop is not None:
            c, d, h, w = img_tensor.shape
            td, th, tw = target_crop
            pad_d = max(0, td - d)
            pad_h = max(0, th - h)
            pad_w = max(0, tw - w)
            img_tensor = F.pad(img_tensor,
                               [pad_w // 2, pad_w - pad_w // 2,
                                pad_h // 2, pad_h - pad_h // 2,
                                pad_d // 2, pad_d - pad_d // 2],
                                mode='constant', value=0)
            img_tensor = img_tensor[:, :td, :th, :tw]  # crop excess if needed

        # Normalize channel-wise (same as training)
        img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, C, D, H, W)
        img_tensor = transform(img_tensor)

        if verbose:
            print(f"Classifying {case_name} with shape {img_tensor.shape}...")

        # Inference
        with torch.no_grad():
            logits = model(img_tensor).squeeze().item()
            prob = torch.sigmoid(torch.tensor(logits)).item()
            label = int(prob >= threshold)

        liver_classifications[case_name] = label
        if verbose:
            print(f"{case_name}: prob={prob:.4f}, label={label}")

    return liver_classifications

def update_liver_mets(lesion_results_dict, 
                        ct_map,
                        pet_map, 
                        organ_dir, 
                        model_path, 
                        device,
                        overwrite,
                        verbose=False):
    
    first_case = next(iter(lesion_results_dict))
    if "liver_mets" in lesion_results_dict[first_case].get("lesion_metrics", {}).get("patient", {}):
        if not overwrite:
            print("Liver mets already classified. Use --overwrite to re-run classification.")
            return

    liver_classifications = classify_liver_disease(ct_map,
                                                    pet_map,
                                                    organ_dir,
                                                    model_path,
                                                    device,
                                                    overwrite,
                                                    verbose=verbose,
                                                )

    total_counts = {0: 0, 1: 0}  # For "All" summary

    for case_name, case_data in lesion_results_dict.items():
        if case_name in ("All", "SUV_threshold"):
            continue

        else:
            # Check segmentation-based liver mets presence
            seg_detected = any(
                lesion.get("ts_name") == "liver"
                for lesion in case_data.get("lesions", {}).values()
            )

            # Get classifier liver mets presence (default to 0 if missing)
            classifier_detected = bool(liver_classifications.get(case_name, 0))

            # Final decision: segmentation positive takes priority
            final_call = 1 if seg_detected else int(classifier_detected)

            # Save result
            case_data.setdefault("lesion_metrics", {}).setdefault("patient", {})["liver_mets"] = final_call

            # Track discrepancies
            if seg_detected and not classifier_detected:
                # Print only the notable discrepancy where seg says yes, classifier says no
                print(f"[Warning] Liver mets discrepancy in {case_name}: "
                    f"Segmentation contains liver mets, but the liver classifier has not identified liver disease. Case review is advised.")

        # Count totals
        total_counts[final_call] += 1

    # Add summary counts to "All"
    lesion_results_dict.setdefault("All", {}).setdefault("patient_metrics", {})
    lesion_results_dict["All"]["patient_metrics"]["liver_mets"] = total_counts


def collate_csv_summary_info(lesion_results_dict, lesion_results_dir, 
                            verbose=False, anonymize=False):
    # Filter out summary entries
    cases = [case for case in lesion_results_dict.keys() if case not in ("All", "SUV_threshold")]

    # Function to detect date in case name segments
    def extract_date_from_case(case_name):
        for segment in case_name.split("_"):
            for fmt in ("%Y%m%d", "%Y-%m-%d"):
                try:
                    return datetime.strptime(segment, fmt)
                except ValueError:
                    continue
        return None  # No date found

    # Prepare anonymization mapping if needed
    anonymized_map = {}
    if anonymize:
        case_counter = 1
        num_digits = len(str(len(cases)))  # zero-padding width
        for case in cases:
            date_obj = extract_date_from_case(case)
            if date_obj:
                anon_name = date_obj.strftime("%Y%m%d")
            else:
                anon_name = str(case_counter).zfill(num_digits)
                case_counter += 1
            anonymized_map[case] = anon_name

    # Metrics column names
    columns = ["Case", "Number_of_lesions", "TTV", "Tumour_SUVmean", "Tumour_SUVmax", "TLU", "TLQ", "Bone_mets", "Visceral_mets", "Liver_mets"]

    rows = []
    for case in cases:
        case_display = anonymized_map[case] if anonymize else case
        case_data = lesion_results_dict[case]

        number_lesions = case_data.get("lesion_metrics", {}).get("region", {}).get("whole_body", {}).get("lesion_count", 0)

        ttv = case_data.get("lesion_metrics", {}).get("region", {}).get("whole_body", {}).get("total_burden", 0)
        ttv = float(f"{ttv:.4g}") if isinstance(ttv, (int, float)) else ttv

        suvmean = case_data.get("lesion_metrics", {}).get("patient", {}).get("SUVmean", 0)
        suvmax = case_data.get("lesion_metrics", {}).get("patient", {}).get("SUVmax", 0)
        tlu = case_data.get("lesion_metrics", {}).get("patient", {}).get("TLU", 0)
        tlq = case_data.get("lesion_metrics", {}).get("patient", {}).get("TLQ", 0)
        # Round to 4 significant figures
        suvmean = float(f"{suvmean:.4g}") if isinstance(suvmean, (int, float)) else suvmean
        suvmax = float(f"{suvmax:.4g}") if isinstance(suvmax, (int, float)) else suvmax
        tlu = float(f"{tlu:.4g}") if isinstance(tlu, (int, float)) else tlu
        tlq = float(f"{tlq:.4g}") if isinstance(tlq, (int, float)) else tlq

        bone_lesion_count = case_data.get("lesion_metrics", {}).get("region", {}).get("bone", {}).get("lesion_count", 0)
        bone_mets = bool(bone_lesion_count > 0)

        visceral_lesion_count = case_data.get("lesion_metrics", {}).get("region", {}).get("visceral", {}).get("lesion_count", 0)
        visceral_mets = bool(visceral_lesion_count > 0)

        liver_mets = bool(case_data.get("lesion_metrics", {}).get("patient", {}).get("liver_mets", 0))

        rows.append([case_display, number_lesions, ttv, suvmean, suvmax,  tlu, tlq, bone_mets, visceral_mets, liver_mets, extract_date_from_case(case)])
    # Sort by date if present
    rows.sort(key=lambda x: (x[10] is None, x[10]))  # None dates go last (date is at index 10)
    rows = [[case, number_lesions, ttv, suvmean, suvmax, tlu, tlq, bone_mets, visceral_mets, liver_mets] for case, number_lesions, ttv, suvmean, suvmax, tlu, tlq, bone_mets, visceral_mets, liver_mets, _, in rows]

    # Save mapping if anonymizing
    if anonymize:
        map_path = os.path.join(lesion_results_dir, "anonymization_map.csv")
        with open(map_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["anon_ID", "case_name"])
            for orig, anon in anonymized_map.items():
                writer.writerow([anon, orig])
        if verbose:
            print(f"Anonymization map saved to {map_path}")
        csv_name = "biomarker_info_anon.csv"
    else:
        csv_name = "biomarker_info.csv"

        csv_path = os.path.join(lesion_results_dir, csv_name)
        # Write CSV
    os.makedirs(lesion_results_dir, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    if verbose:
        print(f"Biomarker info saved to {csv_path}")


## Main post-processing function ##
def post_process(
        list_of_lists_prepro, # list of [ct_path, pet_path] pairs identified for preprocessing
        output_pred_dir, # location of model's predictions
        organ_dir, # location of organ segmentations
        liver_model_path, # path to liver disease classification model
        rtstruct_processing, # whether to do RTSTRUCT conversion
        ct_dicom_case_map, # for RTSTRUCT conversion (requires original DICOM)
        device,
        suv_thresh,
        exp_segs,
        fast,
        verbose,
        overwrite,
        anonymize,
    ):
    # Build ct_map and pet_map from list_of_lists
    ct_map = {}
    pet_map = {}
    for pair in list_of_lists_prepro:
        ct_path = Path(pair[0])
        pt_path = Path(pair[1])
        # Robustly remove .nii.gz or .nii and _0000 suffix
        name = ct_path.name
        if name.endswith('.nii.gz'):
            ct_base = name[:-7]  # remove .nii.gz
        elif name.endswith('.nii'):
            ct_base = name[:-4]  # remove .nii
        else:
            raise ValueError(f"Unexpected CT file format: {ct_path.name}")
        if '_0000' in ct_base:
            ct_base = ct_base.rsplit('_0000', 1)[0]
        elif 'CT' in ct_base:
            ct_base = ct_base.rsplit('CT', 1)[0]
        # print(f"CT base: {ct_base}")

        case_base = ct_base
        ct_map[case_base] = str(ct_path)
        pet_map[case_base] = str(pt_path)
        # print(f"Mapped case {case_base}: CT={ct_path}, PET={pt_path}")

    # Derive output_base from output_pred_dir (used for all output dirs/files)
    output_base = Path(output_pred_dir).name.replace('_outputs', '')
    output_parent = str(Path(output_pred_dir).parent)

    # Define all output dirs/files using output_base
    rtstruct_dir = os.path.join(output_parent, f"{output_base}_output_rtstructs")
    organ_dir_default = os.path.join(output_parent, f"{output_base}_organ_segmentations")
    lesion_results_dir = os.path.join(output_parent, f"{output_base}_lesion_classification")
    if exp_segs:
        lesion_results_dir += "_expanded"

    # SUV THRESHOLDING
    if suv_thresh > 0:
        apply_suv_threshold(pet_map,
                            output_pred_dir,
                            suv_thresh,
                            verbose,
                            overwrite)
        logging.info(f"Applied SUV thresholding (threshold = {suv_thresh}) to segmentation outputs in {output_pred_dir}")
        lesion_results_json = f"lesion_results_suv_thresh_{int(suv_thresh)}.json"
    else:
        logging.info(f"No SUV thresholding applied (threshold = {suv_thresh}) to segmentation outputs.")
        lesion_results_json = "lesion_results.json"

    # RTSTRUCT CONVERSION
    if rtstruct_processing:
        os.makedirs(rtstruct_dir, exist_ok=True)
        logging.info(f"Converting segmentations to RTSTRUCT format and saving to {rtstruct_dir}")
        seg_to_rtstruct(output_pred_dir,
                        ct_dicom_case_map,
                        ct_map,
                        rtstruct_dir,
                        verbose,
                        overwrite)

    # ORGAN SEGMENTATION
    if organ_dir is None:
        organ_dir = organ_dir_default
        logging.info(f"No organ segmentation directory specified. Using default: {shorten_path(organ_dir)}")
    else:
        logging.info(f"Using user-specified organ segmentation directory: {shorten_path(organ_dir)}")
    os.makedirs(organ_dir, exist_ok=True)
    # Generate organ segmentations for each case in ct_map
    generate_organ_segmentations(ct_map, organ_dir,
                                    device, fast, verbose)
    
    # SEGMENTATION EXPANSION
    if exp_segs:
        logging.info(f"Expanding segmentation outputs in {output_pred_dir}")
        output_pred_dir_expanded = expand_segmentations(
                                        lesion_dir=output_pred_dir,
                                        organ_dir=organ_dir,
                                        ct_map=ct_map,
                                        pet_map=pet_map,
                                        overwrite=overwrite,
                                        verbose=verbose
                                    )
    else:
        output_pred_dir_expanded = None # Not used if not expanding

    # LESION CLASSIFICATION (and metrics: iterate over cases in output_pred_dir (or output_pred_dir_expanded if used))
    os.makedirs(lesion_results_dir, exist_ok=True)
    lesion_results_json_path = os.path.join(lesion_results_dir, lesion_results_json)
    if not overwrite and os.path.exists(lesion_results_json_path):
        logging.info(f"Lesion results JSON already exists at {shorten_path(lesion_results_json_path)} and overwrite is False.")
        # Load existing lesion results
        with open(lesion_results_json_path, 'r') as f:
            lesion_results_dict = json.load(f)
        logging.info(f"Loaded existing lesion results from {lesion_results_json_path}")
    else:
        if verbose:
            logging.info(f"Lesion results JSON will be saved to {shorten_path(lesion_results_json_path)}")
        # Classify lesions (using generated organ segs) and extract biomarkers
        if output_pred_dir_expanded is not None:
            lesion_dir_to_classify = output_pred_dir_expanded
            logging.info(f"Classifying lesions from expanded segmentations in {shorten_path(output_pred_dir_expanded)}")
        else:
            lesion_dir_to_classify = output_pred_dir
        lesion_results_dict = lesion_classifier(
                                lesion_dir=lesion_dir_to_classify,
                                organ_dir=organ_dir,
                                pet_map=pet_map,
                                verbose=verbose,
                                case_filter=None,
                                overlap_threshold=0.5,
                                organ_suffix="_total"
                            )
        lesion_results_dict["SUV_threshold"] = suv_thresh

    # LIVER DISEASE CLASSIFICATION
    # liver_model_path = '/media/joel/Pal_CT/PSMA-PET/Models/Liver_classifier/full_train/exp1_high_alpha/run_20250704-190755/best_model.pth'
    # Update lesion results with liver mets classification
    update_liver_mets(lesion_results_dict,
                        ct_map=ct_map,
                        pet_map=pet_map,
                        organ_dir=organ_dir,
                        model_path=liver_model_path,
                        device=device,
                        overwrite=overwrite,
                        verbose=verbose)

    # CSV SUMMARY CONSTRUCTION
    collate_csv_summary_info(lesion_results_dict, lesion_results_dir,
                            verbose=verbose,
                            anonymize=anonymize)

    with open(lesion_results_json_path, 'w') as f:
        json.dump(lesion_results_dict, f, indent=4)
    logging.info(f"Saved lesion classification and metrics json to {lesion_results_json_path}")

    return