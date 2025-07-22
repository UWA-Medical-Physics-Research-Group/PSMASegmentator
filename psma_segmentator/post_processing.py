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
from psma_segmentator.pre_processing import shorten_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

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


def expand_segmentation(predicted_image, pet_image, 
                                ct_image_dir = str, suv_threshold=3):
    """
    Expand the predicted segmentation based on the PET image and a given SUV threshold.

    :param predicted_image: SimpleITK image of the predicted segmentation.
    :param pet_image: SimpleITK image of the PET image.
    :param ct_image: SimpleITK image of the CT image.
    :param suv_threshold: The SUV threshold to expand the segmentation by. Defaults to three. 
    :return: The expanded segmentation.
    """
    # Ensure the PET image is of type sitkFloat32
    pet_image = sitk.Cast(pet_image, sitk.sitkFloat32)

    # Label the connected components in the predicted segmentation
    labeled_image = sitk.ConnectedComponent(predicted_image)

    # Get the number of connected components
    num_components = int(sitk.GetArrayViewFromImage(labeled_image).max())

    # Create an empty image to store the expanded segmentation
    expanded_seg = sitk.Image(predicted_image.GetSize(), sitk.sitkUInt8)
    expanded_seg.CopyInformation(predicted_image)

    #Run the TotalSegmentator on the input CT image
    ct_segmentation_nib = totalsegmentator(ct_image_dir,
                                            fastest = True)
    
    #Convert the nibabel image to a NumPy array
    ct_segmentation_array = ct_segmentation_nib.get_fdata()

    #Convert to int
    ct_segmentation_array = ct_segmentation_array.astype(int)

    #Just get the components of the CT segmentation equal to 5 or 21 or 22 (liver, bladder, prostate)
    ct_bladder_prostate = np.isin(ct_segmentation_array, [5, 21, 22]).astype(np.uint8)
    ct_bladder_prostate = np.transpose(ct_bladder_prostate, (2, 1, 0)) #Converting from (z, y, x) to (x, y, z) format

    # Iterate over each connected component
    for i in range(1, num_components + 1):
        # Get the current component
        component = labeled_image == i

        # Find all seed points for the current connected component
        component_array = sitk.GetArrayFromImage(component)
        pet_array = sitk.GetArrayFromImage(pet_image)

        # Find the voxel with the maximum SUV value within the component
        component_voxels = np.argwhere(component_array)
        max_voxel = max(component_voxels, key=lambda x: pet_array[tuple(x)])
        seed_point = tuple(max_voxel[::-1])  # Convert to (z, y, x) format
        seed_point = (int(seed_point[0]), int(seed_point[1]), int(seed_point[2]))
        expanded_component = sitk.ConnectedThreshold(image1=pet_image,
                                                        seedList=[seed_point],
                                                        lower=suv_threshold,
                                                        upper=1000.0,
                                                        replaceValue=1)
        
        #Get the array from the expanded component
        expanded_component_array = sitk.GetArrayFromImage(expanded_component)

        #Implement the rubric for selecting which components are expanded. Check if the expanded component
        #just created overlaps with the urinary bladder of the ct segmentation (value is 21), or if the expanded
        #component overlaps with the prostate (value is 22). If it does, then don't expand the component.
        #If it doesn't, then expand the component.
        
        #If any intersection between the expanded component and the liver/bladder/prostate is found, 
        # then don't expand the component.
        if np.any(np.logical_and(expanded_component_array, ct_bladder_prostate)):
            expanded_seg = sitk.Or(expanded_seg, component)
        
        #Handle the case where the component is all zeroes because SUVmax lower than threshold
        elif np.all(expanded_component_array == 0):
            expanded_seg = sitk.Or(expanded_seg, component)
        
        else:
            expanded_seg = sitk.Or(expanded_seg, expanded_component)
            expanded_seg = sitk.Or(expanded_seg, component)
        
    return expanded_seg


## Generate organ segmentations using TotalSegmentator ##
def generate_organ_segmentations(prepro_dir, output_segs_dir, 
                                    device, 
                                    fast,
                                    verbose):
    if device == 'cuda':
        device = 'gpu'  # update to 'gpu' for compatibility with TotalSegmentator

    cases = [case for case in os.listdir(prepro_dir) if case.endswith("0000.nii.gz")]

    for case in cases:
        case_base = case.replace("_0000.nii.gz", "")
        case_path = os.path.join(prepro_dir, case)
        out_path_total = os.path.join(output_segs_dir, f"{case_base}_total.nii.gz")

        # Check for existing *_total output
        if os.path.exists(out_path_total):
            print(f"Organ segmentation already exists for {shorten_path(case_path)} at {shorten_path(out_path_total)}. Skipping.")
            continue
        else:
            if verbose:
                print(f"Saving organ segmentation for {shorten_path(case_path)} to {shorten_path(out_path_total)}")

        # Look for any existing seg for this case
        matching_segs = [f for f in os.listdir(output_segs_dir) if f.startswith(case_base) and f.endswith(".nii.gz")]
        valid_total_found = False

        for seg_file in matching_segs:
            if "_total" not in seg_file:
                seg_path = os.path.join(output_segs_dir, seg_file)
                try:
                    seg_img = nib.load(seg_path)
                    seg_data = seg_img.get_fdata()
                    unique_labels = np.unique(seg_data)
                    if (len(unique_labels) == 117) or (unique_labels.max() == 117):
                        # This is a total segmentation, but misnamed
                        new_path = os.path.join(output_segs_dir, f"{case_base}_total.nii.gz")
                        os.rename(seg_path, new_path)
                        print(f"Found valid TotalSegmentator output for {case_base} without _total suffix. Renamed to: {shorten_path(new_path)}")
                        valid_total_found = True
                        break  # No need to generate again
                except Exception as e:
                    print(f"Could not check file {seg_path}: {e}")

        if valid_total_found:
            continue

        # Otherwise, run TotalSegmentator
        print(f"Processing: {case_path}")
        command = f"TotalSegmentator -i '{case_path}' -o '{out_path_total}' --ta total --ml -d {device}"
        if fast:
            command += " --fast"
            if verbose:
                print("Running TotalSegmentator in 'fast' mode.")
        os.system(command)


## Apply SUV threshold to segmentation outputs ##
def apply_suv_threshold(prepro_dir, output_dir, 
                        suv_thresh, verbose,
                        overwrite):
    seg_files = [f for f in os.listdir(output_dir) if f.endswith('.nii.gz')]

    for seg_file in seg_files:
        seg_path = os.path.join(output_dir, seg_file)
        if verbose:
            print(f"Processing segmentation file: {shorten_path(seg_path)}")
        seg_img = nib.load(seg_path)
        seg_data = seg_img.get_fdata()

        seg_base = os.path.splitext(os.path.splitext(seg_file)[0])[0]
        pt_path = os.path.join(prepro_dir, seg_base + '_0001.nii.gz')
        if not os.path.exists(pt_path):
            print(f"Warning: Corresponding PET image not found for {shorten_path(seg_file)} in {shorten_path(pt_path)}. Skipping.")
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
            backup_dir = os.path.join(output_dir, "backups_no_threshold")
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

    suv_values = pet_array[mask_array]
    suv_mean = np.mean(suv_values)
    suv_max = np.max(suv_values)
    suv_total = suv_mean * np.sum(mask_array) * voxel_volume_ml

    return {
        'SUVmean': round_sig(suv_mean, 6),
        'SUVmax': round_sig(suv_max, 6),
        'SUVtotal': round_sig(suv_total, 6)
    }

## Lesion classification and metric calculation ##
def calc_lesion_volume(pt_img, mask_img):
    """
    Calculates volume of a binary lesion mask in cubic centimeters (cm³).

    Inputs:
        pt_img (nibabel.Nifti1Image): PET or CT image, used to get voxel spacing
        mask_img (np.ndarray): Binary mask array (1 = lesion, 0 = background)

    Returns:
        float: Lesion volume in cm³
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

    Returns:
        chosen_class (str): The assigned class (organ name or nodal classification).
        method_used (str): One of 'overlap', 'binary classification', or 'distance'.
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
            raise ValueError("CIB z-slice not found - assuming iliac artery segmentations not provided. Cannot classify nodal lesion.")
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
            "ts_code": ts_code.strip(),
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
                "SUVtotal": suv_metrics.get("SUVtotal", 0)
            }

    return case_dict

def lesion_classifier(lesion_dir, # from output_dir
                        organ_dir, # newly created organ_dir
                        img_dir, # from input_dir
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

        if img_dir:
            pt_path = os.path.join(img_dir, f"{case_name}_0001.nii.gz")
            if not os.path.exists(pt_path):
                logging.warning(f"PT path does not exist: {pt_path}")
            else:
                pt_img = nib.load(pt_path)
        else:
            pt_img = None

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


## Main post-processing function ##
def post_process(
        prepro_dir, # location of NIfTI images (either input_path or newly created _preprocessed path)
        output_dir, # location of model's predictions
        organ_dir, # location of organ segmentations
        device,
        suv_thresh,
        fast,
        verbose,
        overwrite
    ):
    if suv_thresh > 0:
        # Apply SUV thresholding
        apply_suv_threshold(prepro_dir, output_dir, 
                            suv_thresh, verbose,
                            overwrite)
        logging.info(f"Applied SUV thresholding (threshold = {suv_thresh}) to segmentation outputs in {output_dir}")
        lesion_results_json = f"lesion_results_suv_thresh_{int(suv_thresh)}.json"
    else:
        logging.info(f"No SUV thresholding applied (threshold = {suv_thresh}) to segmentation outputs.")
        lesion_results_json = "lesion_results.json"
    
    if organ_dir is None:
        organ_dir = os.path.join(os.path.dirname(output_dir), "organ_segmentations")
        os.makedirs(organ_dir, exist_ok=True)
        logging.info(f"No folder containing organ segmentations specified, newly generated segmentations will be saved to {shorten_path(organ_dir)}")
    else:
        logging.info(f"Using organ segmentations from {shorten_path(organ_dir)}")

    # Generate organ segmentations
    generate_organ_segmentations(prepro_dir, organ_dir, 
                                    device,
                                    fast, 
                                    verbose)

    lesion_results_dir = os.path.join(Path(output_dir).parent, "lesion_classification")
    if not os.path.exists(lesion_results_dir):
        os.makedirs(lesion_results_dir, exist_ok=True)
        logging.info(f"Created lesion classification directory at {shorten_path(lesion_results_dir)}")
    lesion_results_json_path = os.path.join(lesion_results_dir, lesion_results_json)
    if not overwrite and os.path.exists(lesion_results_json_path):
        logging.info(f"Lesion results JSON already exists at {shorten_path(lesion_results_json_path)} and overwrite is False. Skipping lesion classification and metrics extraction.")
        return
    else:
        logging.info(f"Lesion results JSON will be saved to {shorten_path(lesion_results_json_path)}")

    # Classify lesions (using generated organ segs) and extract biomarkers
    lesion_results_dict = lesion_classifier(
                            lesion_dir=output_dir,
                            organ_dir=organ_dir,
                            img_dir=prepro_dir,
                            verbose=verbose,
                            case_filter=None,
                            overlap_threshold=0.5,
                            organ_suffix="_total"
                        )

    lesion_results_dict["SUV_threshold"] = suv_thresh

    with open(lesion_results_json_path, 'w') as f:
        json.dump(lesion_results_dict, f, indent=4)
    logging.info(f"Saved lesion classification and metrics json to {lesion_results_json_path}")

    return