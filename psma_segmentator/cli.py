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
        help="Directory to save segmentation results. Defaults to .../input_dir.parent/outputs."
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
            )

if __name__ == "__main__":
    main()