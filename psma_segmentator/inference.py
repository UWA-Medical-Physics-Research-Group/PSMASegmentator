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


def segmentate(model_folder, list_of_lists, 
                output_dir, 
                device, 
                use_tta,
                verbose, 
                step_size=0.5):
    """
    Runs inference using nnUNet for all cases in the preprocessed directory.
    """
    assert device in ['cuda', 'cpu', 'mps'] or isinstance(device, torch.device), "Invalid device specified."
    
    if device == 'cpu':
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
        perform_everything_on_device = False
    elif device == 'cuda':
        torch.set_num_threads(1)
        device = torch.device('cuda')
        perform_everything_on_device = True
    else:
        device = torch.device('mps')

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
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=None, checkpoint_name="checkpoint_final.pth")

    # Run nnUNet inference on the entire preprocessed directory
    print(f"\nRunning prediction on {list_of_lists}...")
    print(f"Saving predictions to output directory: {output_dir}")
    predictor.predict_from_files(
        list_of_lists_or_source_folder=list_of_lists,
        output_folder_or_list_of_truncated_output_files=output_dir,
        save_probabilities=False,
        overwrite=True
    )