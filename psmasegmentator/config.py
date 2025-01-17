import os
from pathlib import Path
from importlib.metadata import version
__version__ = version("psma-segmentator")


def get_psmasegmentator_dir(version):
    """
    Get the path to the psmasegmentator directory, containing the model weights
    that have been downloaded.
    """
    
    if "PSMA_SEGMENTATOR_HOME" in os.environ:
        base_dir = Path(os.environ["PSMA_SEGMENTATOR_HOME"])
    else:
        base_dir = Path.home() / ".psmasegmentator"
    return base_dir / version

def setup_psmasegmentator():
    current_version = __version__
    print(f"Setting up psmasegmentator version {current_version}")
    version_dir = get_psmasegmentator_dir(current_version)
    weights_dir = version_dir / "results"
    weights_dir.mkdir(parents=True, exist_ok=True)

    os.environ["nnUNet_raw"] = str(weights_dir)
    os.environ["nnUNet_preprocessed"] = str(weights_dir)
    os.environ["nnUNet_results"] = str(weights_dir)
    return weights_dir
