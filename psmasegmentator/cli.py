import argparse

### PACKAGE IMPORTS
#from psma_segmentator.dicom_processing import process_dicom
#from psma_segmentator.nifti_processing import process_nifti
#from psma_segmentator.segmentation import run_segmentation

def main():
    parser = argparse.ArgumentParser(description="PSMA PET Auto-Segmentation Tool")
    parser.add_argument("--input_dir", required=True, help="Directory containing PSMA PET and CT files")
    parser.add_argument("--file_format", choices=["dicom", "nifti"], required=True, help="Input file format")
    parser.add_argument("--output_dir", required=True, help="Directory to save segmentation results")
    
    args = parser.parse_args()
    
    if args.file_format == "dicom":
        images = process_dicom(args.input_dir)
    elif args.file_format == "nifti":
        images = process_nifti(args.input_dir)
    else:
        raise ValueError("Unsupported file format")
    
    run_segmentation(images, args.output_dir)

if __name__ == "__main__":
    main()