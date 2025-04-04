import argparse
import torch
from psma_segmentator.python_api import psma_segmentator

def main():
    parser = argparse.ArgumentParser(description="PSMA PET/CT Auto-Segmentation Tool.")
    
    parser.add_argument(
        "-i", "--input_dir", required=True, help="Path to directory containing PET/CT files."
    )
    parser.add_argument(
        "-p", "--personal_access_token", required = True, help= "GitHub Personal Access Token (PAT) for downloading weights."
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Directory to save segmentation results."
    )
    parser.add_argument(
        "-f", "--file_format", choices=["dicom", "nifti"], required=False, help="Input file format - Use 'dicom' for DICOM files, and 'nifti' for .nii or .nii.gz files."
    )
    parser.add_argument(
        "-d", "--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for processing. Defaults to 'cuda' if available, otherwise 'cpu'."
    )    
    args = parser.parse_args()

    psma_segmentator(
                input_dir = args.input_dir, 
                token = args.personal_access_token, 
                output_dir = args.output_dir, 
                # file_format= args.file_format,
                device = args.device
            )

if __name__ == "__main__":
    main()