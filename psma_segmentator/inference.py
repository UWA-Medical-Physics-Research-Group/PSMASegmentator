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


def segmentate(model_folder, list_of_lists, output_dir, 
                device, step_size=0.5, use_tta=False, verbose=False):
    """
    Runs inference using nnUNet for all cases in the preprocessed directory.
    """
    assert device in ['cuda', 'cpu', 'mps'] or isinstance(device, torch.device), "Invalid device specified."
    
    if device == 'cpu':
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif device == 'cuda':
        torch.set_num_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    # Initialize predictor
    predictor = nnUNetPredictor(
        tile_step_size=step_size,
        use_gaussian=True,
        use_mirroring=use_tta,
        perform_everything_on_device=True, 
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