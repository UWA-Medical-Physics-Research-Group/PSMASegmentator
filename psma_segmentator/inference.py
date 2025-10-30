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
import tempfile
from pathlib import Path
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import torch
import multiprocessing
import pydicom
from datetime import datetime
import math
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def segmentate(model_folder,
                list_of_lists_pred,
                output_pred_dir,
                device,
                use_tta,
                verbose,
                step_size=0.5,
                checkpoint_name: str = "checkpoint_final.pth"):
    """
    Runs inference using nnUNet for all cases in the preprocessed directory.
    """    
    if device == 'cpu':
        torch.set_num_threads(multiprocessing.cpu_count())
        perform_everything_on_device = False
    else: # assuming 'cuda' or 'cuda:n'
        torch.set_num_threads(1)
        perform_everything_on_device = True
    device = torch.device(device) # convert str to torch device

    # Initialize predictor
    predictor = nnUNetPredictor(
        tile_step_size=step_size,
        use_gaussian=True,
        use_mirroring=use_tta,
        perform_everything_on_device=perform_everything_on_device, 
        device=device,
        verbose=verbose,
        allow_tqdm=True
    )
    # Initialize predictor from the requested checkpoint. Pass checkpoint_name to allow
    # choosing 'checkpoint_final.pth' (default) or 'checkpoint_best.pth' (or other name).
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=None, checkpoint_name=checkpoint_name)

    # Run nnUNet inference on the entire preprocessed directory
    print(f"\nRunning prediction on {list_of_lists_pred}...")
    if list_of_lists_pred is not None:
        print(f"Saving predictions to output directory: {output_pred_dir}")
    print("Inference parameters: ")
    print(f" - Model folder: {model_folder}")
    print(f" - Checkpoint name: {checkpoint_name}")
    print(f" - Device: {device}")
    print(f" - Use TTA: {use_tta}")
    print(f" - Tile step size: {step_size}")

    # Run prediction
    predictor.predict_from_files(
        list_of_lists_or_source_folder=list_of_lists_pred,
        output_folder_or_list_of_truncated_output_files=output_pred_dir,
        save_probabilities=False,
        overwrite=True
    )