import os
from pathlib import Path
import requests


def nnunet_setup()
    """
    Set up the required paths to run nnunet.
    """

    weights_dir = ## SET the path to the weights directory
    os.environ["nnUNet_raw"] = str(weights_dir) #Should not be needed for running inference.
    os.environ["nnUNet_preprocessed"] = str(weights_dir) #Should not be needed for running inference.
    os.environ["nnUNet_results"] = str(weights_dir)
