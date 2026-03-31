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
import sys
from pathlib import Path
from datetime import datetime

def main():
    print("GREETINGS, PROGRAM. Welcome to the Digital Frontier of PSMA PET/CT Segmentation...\n")

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
        "-chkpt", "--checkpoint_name", required=False, type=str, default="checkpoint_final.pth",
        help="Name of the checkpoint file to use for inference."
    )
    parser.add_argument(
        "-plans", "--plans_name", required=False, type=str, default="plans.json",
        help="Name of the plans file to use for inference."
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
        "-exp_segs", "--expand_segmentations", required=False, action="store_true",
        help="Expand segmentations during post-processing."
    )
    parser.add_argument(
        "-or", "--organ_dir", required=False, default=None,
        help="Directory containing organ segmentations for post-processing lesion classification. Defaults to .../output_dir.parent/organ_segmentations."
    )
    parser.add_argument(
        "--fast", required=False, action="store_true",
        help="Use fast mode for inference. This uses the Fast (lightweight) version of PSMASegmentator, disables Test-Time Augmentation (TTA), and uses the --fast flag in TotalSegmentator for faster organ segmentation generation."
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
    parser.add_argument(
        "--save_log", required=False, action="store_true",
        help=("If set, save the entire CLI stdout/stderr to a timestamped .txt file in the "
                "parent directory of --output_dir (or parent of --input_dir, or cwd if neither)."),
    )

    args = parser.parse_args()

    # Setup optional CLI logging (tee stdout/stderr to a file) early so all messages are captured
    log_file = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if getattr(args, "save_log", False):
        # Determine parent directory for the log file
        if args.output_dir:
            # Place log next to the output directory (parent of output_dir)
            log_parent = Path(args.output_dir).parent
        elif args.input_dir:
            # Save logs in a dedicated subdirectory named '<input_dir_name>_logs'
            input_path = Path(args.input_dir)
            log_parent = input_path.parent / f"{input_path.name}_logs"
        else:
            log_parent = Path.cwd()
        try:
            log_parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            log_parent = Path.cwd()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_parent / f"psma_segmentator_run_{timestamp}.txt"

        try:
            log_file = open(log_path, "w", encoding="utf-8")
        except Exception as e:
            print(f"[WARNING] Could not open log file {log_path} for writing: {e}")
            log_file = None

        if log_file is not None:
            class Tee:
                def __init__(self, a, b):
                    self.a = a
                    self.b = b
                def write(self, data):
                    try:
                        self.a.write(data)
                    except Exception:
                        pass
                    try:
                        self.b.write(data)
                    except Exception:
                        pass
                def flush(self):
                    try:
                        self.a.flush()
                    except Exception:
                        pass
                    try:
                        self.b.flush()
                    except Exception:
                        pass

            sys.stdout = Tee(original_stdout, log_file)
            sys.stderr = Tee(original_stderr, log_file)
            print(f"CLI output is being saved to: {log_path}")

    # If 'torch' is already imported, warn the user. Changing CUDA_VISIBLE_DEVICES after
    # torch has initialized may not have any effect. We do NOT force a CPU fallback here;
    # instead we print a clear warning so the user can restart the process if they need
    # a different CUDA device configuration.
    if 'torch' in sys.modules:
        print(
            "[WARNING] The 'torch' module is already imported in this Python process.\n"
            "CUDA may already be initialized in this process. If you requested a specific "
            "GPU with --device (for example 'cuda:1') and it doesn't appear to be used, "
            "please restart the process so the requested device can be honoured before "
            "any import of torch.",
            file=sys.stderr,
        )

    # Respect whatever device the user requested. Users expect that passing an absolute
    # CUDA ordinal such as 'cuda:23' will use the system-wide device with that ordinal.
    # Do not remap or override CUDA_VISIBLE_DEVICES here.
    if args.device is not None:
        if re.match(r"^cuda:\d+$", args.device):
            # Informational message only; we do not change environment vars or remap.
            print(f"[INFO] Using requested device '{args.device}'. This uses the system-wide CUDA ordinal.")

    # import here so CUDA_VISIBLE_DEVICES is set before any torch import inside psma_segmentator
    try:
        from psma_segmentator.python_api import psma_segmentator

        psma_segmentator(
                input_dir = args.input_dir, 
                input_ct = args.input_ct,
                input_pet = args.input_pet,
                output_pred_dir = args.output_dir,
                weights_dir = args.weights_dir,
                checkpoint_name = args.checkpoint_name,
                plans_name= args.plans_name,
                token = args.personal_access_token,
                version= args.version,
                device = args.device,
                rtstruct_processing = args.rtstruct_processing,
                preprocess_only = args.preprocess_only,
                disable_postprocessing = args.disable_postprocessing,
                suv_thresh = args.suv_threshold,
                exp_segs = args.expand_segmentations,
                organ_dir = args.organ_dir,
                fast = args.fast,
                show_w = args.show_w,
                show_c = args.show_c,
                anonymize = args.anonymize,
                overwrite = args.overwrite,
                verbose = args.verbose,
            )
    finally:
        # Restore stdout/stderr and close log file if we opened one
        if getattr(args, "save_log", False) and log_file is not None:
            try:
                sys.stdout = original_stdout
            except Exception:
                pass
            try:
                sys.stderr = original_stderr
            except Exception:
                pass
            try:
                log_file.close()
            except Exception:
                pass
if __name__ == "__main__":
    main()