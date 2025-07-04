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
import torch
from psma_segmentator.python_api import psma_segmentator

def main():
    parser = argparse.ArgumentParser(description="PSMA PET/CT Auto-Segmentation Tool.")
    
    parser.add_argument(
        "-i", "--input_dir", required=True, 
        help="Path to directory containing PET/CT files."
    )
    parser.add_argument(
        "-o", "--output_dir", required=False, default=None,
        help="Directory to save segmentation results. Defaults to .../input_dir.parent/input_dir.name_outputs."
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
        "-d", "--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for processing. Defaults to 'cuda' if available, otherwise 'cpu'."
    )
    parser.add_argument(
        "--include_rtstructs", required=False, action="store_true",
        help="Specify if RTSTRUCTs (in DICOM form) should be pre-processed, if present. Defaults to False."
    )        
    parser.add_argument(
        "-v", "--verbose", required=False, action="store_true",
        help="Enable verbose output."
    )
    parser.add_argument(
        "-f", "--force", dest="overwrite", required=False, action="store_true",
        help="Overwrite existing pre-processing and segmentation results."
    )
    parser.add_argument(
        "-ppo", "--preprocess_only", required=False, action="store_true",
        help="Pre-process the input files only. No segmentation will be performed."
    )
    parser.add_argument(
        "-pso", "--postprocess_only", required=False, action="store_true",
        help="Post-process the (expected) output files only. No pre-processing or segmentation will be performed."
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

    parser.add_argument(
        "--show_w", action="store_true", help="Show the GNU General Public License warranty disclaimer."
    )

    parser.add_argument(
        "--show_c", action="store_true", help="Show the GNU General Public License terms and conditions."
    )

    args = parser.parse_args()

    psma_segmentator(
                input_dir = args.input_dir, 
                output_dir = args.output_dir, 
                token = args.personal_access_token, 
                version= args.version,
                device = args.device,
                incl_rtstructs = args.include_rtstructs,
                verbose = args.verbose,
                overwrite = args.overwrite,
                preprocess_only = args.preprocess_only,
                postprocess_only = args.postprocess_only,
                suv_thresh = args.suv_threshold,
                organ_dir = args.organ_dir,
                fast = args.fast,
                show_w = args.show_w,
                show_c = args.show_c
            )

if __name__ == "__main__":
    main()