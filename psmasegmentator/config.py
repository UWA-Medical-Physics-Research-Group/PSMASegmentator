import os
from pathlib import Path


def get_psmasegmentator_dir():
    """
    Get the path to the psmasegmentator directory, containing the model weights
    that have been downloaded.
    """
    
    if "PSMA_SEGMENTATOR_DIR" in os.environ:
        return Path(os.environ["PSMA_SEGMENTATOR_DIR"])
    else:
        # Put in a temporary directory for now
        return Path("/tmp/.psmasegmentator")

def nnunet_setup():
    """
    Set up the required paths to run nnunet.
    """
    psma_segmentator_dir = get_psmasegmentator_dir()
    weights_dir = psma_segmentator_dir / "weights"

    os.environ["nnUNet_raw"] = str(weights_dir) #Should not be needed for running inference.
    os.environ["nnUNet_preprocessed"] = str(weights_dir) #Should not be needed for running inference.
    os.environ["nnUNet_results"] = str(weights_dir)
