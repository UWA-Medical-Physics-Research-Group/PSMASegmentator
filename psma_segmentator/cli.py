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

import argparse
import re
import os

def main():
    parser = argparse.ArgumentParser(description="PSMA PET/CT Auto-Segmentation Tool.")
    
    parser.add_argument(
        "-i", "--input_dir", required=False, default=None,
        help="Path to single directory containing PET/CT files in DICOM or NIfTI format."
    )
    parser.add_argument(
        "-i_ct", "--input_ct", required=False, default=None,
        help="Path to either a single CT NIfTI file or a directory containing CT files. If provided, this will be used instead of the input_dir for CT segmentation."
    )
    parser.add_argument(
        "-i_pet", "--input_pet", required=False, default=None,
        help="Path to either a single PET NIfTI file or a directory containing PET files. If provided, this will be used instead of the input_dir for PET segmentation."
    )
    parser.add_argument(
        "-o", "--output_dir", required=False, default=None,
        help="Directory to save segmentation results. Defaults to .../input_dir.parent/input_dir.name_outputs."
    )
    parser.add_argument(
        "-w", "--weights_dir", required=False, default=None,
        help="Directory to look for existing model weights or to store downloaded model weights. Defaults to ~/.psmasegmentator/[version]."
    )
    parser.add_argument(
        "-pat", "--personal_access_token", required = True, 
        help= "GitHub Personal Access Token (PAT) for downloading weights."
    )
    parser.add_argument(
        "--version", required=False, type=str, default=None,
        help="Specify PSMA Segmentator version to use (in form x.y.z). Defaults to latest release if not provided."
    )
    parser.add_argument(
        "-d", "--device", type=str, required=False, default=None,
        help="Device to use for processing, e.g., 'cpu', 'cuda', or 'cuda:n' (0 <= n <= num_gpus). Defaults to 'cuda' if available, otherwise 'cpu'."
    )
    parser.add_argument(
        "-rts", "--rtstruct_processing", required=False, action="store_true",
        help="If True, will convert any found RTSTRUCTs to NIfTI and convert output NIfTIs to RTSTRUCTs. Defaults to False."
    )        
    parser.add_argument(
        "-ppo", "--preprocess_only", required=False, action="store_true",
        help="Only perform preprocessing and save the preprocessed files. No segmentation will be performed."
    )
    parser.add_argument(
        "-dpp", "--disable_postprocessing", required=False, action="store_true",
        help="Disable post-processing of the output files to just do segmentation."
    )
    parser.add_argument(
        "-suv", "--suv_threshold", required=False, type=float, default=0.0,
        help="Specify SUV threshold to apply to segmentation outputs. Defaults to 0."
    )
    parser.add_argument(
        "-or", "--organ_dir", required=False, default=None,
        help="Directory containing organ segmentations for post-processing lesion classification. Defaults to .../output_dir.parent/organ_segmentations."
    )
    parser.add_argument(
        "--fast", required=False, action="store_true",
        help="Use fast mode for inference. This disables Test-Time Augmentation (TTA), and uses the --fast flag in TotalSegmentator for faster organ segmentation generation."
    )
    parser.add_argument( # not currently in readme
        "-sw", "--show_w", action="store_true", help="Show the GNU General Public License warranty disclaimer."
    )
    parser.add_argument( # not currently in readme
        "-sc", "--show_c", action="store_true", help="Show the GNU General Public License terms and conditions."
    )
    parser.add_argument( # not currently in readme
        "-an", "--anonymize", action="store_true", default=False, help="Anonymize patient-identifiable data in the output results."
    )
    parser.add_argument(
        "-f", "--force", dest="overwrite", required=False, action="store_true",
        help="Overwrite existing pre-processing and segmentation results."
    )
    parser.add_argument(
        "-v", "--verbose", required=False, action="store_true",
        help="Enable verbose output."
    )

    args = parser.parse_args()

    if args.device is not None:
        if re.match(r"^cuda:\d+$", args.device):
            gpu_idx = args.device.split(":")[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
            print(f"Restricted CUDA visibility to GPU {gpu_idx}")

    # import here so CUDA_VISIBLE_DEVICES is set before any torch import inside psma_segmentator
    from psma_segmentator.python_api import psma_segmentator

    psma_segmentator(
                input_dir = args.input_dir, 
                input_ct = args.input_ct,
                input_pet = args.input_pet,
                output_pred_dir = args.output_dir,
                weights_dir = args.weights_dir,
                token = args.personal_access_token,
                version= args.version,
                device = args.device,
                rtstruct_processing = args.rtstruct_processing,
                preprocess_only = args.preprocess_only,
                disable_postprocessing = args.disable_postprocessing,
                suv_thresh = args.suv_threshold,
                organ_dir = args.organ_dir,
                fast = args.fast,
                show_w = args.show_w,
                show_c = args.show_c,
                anonymize = args.anonymize,
                overwrite = args.overwrite,
                verbose = args.verbose,
            )

if __name__ == "__main__":
    main()