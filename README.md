# PSMASegmentator

**PSMASegmentator** is a tool for the automatic segmentation of PSMA PET/CT images using Deep Learning (DL). 

The segmentation model uses the [nnUNetv2 framework](https://github.com/MIC-DKFZ/nnUNet), along with the DKFZ LesionTracer team's winning [autoPET-III methodology](https://github.com/mic-dkfz/autopet-3-submission). 
The training dataset is across multiple institutions and scanner types, with 597 patients from the [autoPET-III dataset](https://autopet-iii.grand-challenge.org/) and 438 from Western Australian institutions, for a total of 1015 patients.

It supports both DICOM and NIfTI inputs and automatically handles pre-processing, inference, and post-processing (including lesion classification and biomarker extraction, utilizing [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)).

---
---

## Installation

It is recommended to use a [Miniconda](https://docs.conda.io/en/latest/miniconda.html) virtual environment.

```bash
conda create -n psma_segmentator python=3.10 -y
conda activate psma_segmentator
```
Then clone the repository and install in editable mode:
```bash
cd ~/Path/to/PSMASegmentator
pip install -e .
```

---

## Usage

Once installed, you can run segmentations using the command-line interface (CLI):

```bash
python -m psma_segmentator.cli -i INPUT_DIR -pat YOUR_TOKEN [options]
```

### Optional Arguments

- `-o`, `--output_dir`  
    Path to save the output `.nii.gz` segmentation files to.  
    **Default:** `.../(input_dir).parent/(input_dir.name)_outputs`

- `--version`  
    Specify the version of the PSMA Segmentator model to use (e.g., `0.0.2`).  
    **Default:** Latest available release.

- `-d`, `--device`  
    Choose the device for inference.  
    **Options:** `"cpu"` or `"cuda"`  
    **Default:** `"cuda"` if available.

- `--include_rtstructs`  
    If present, will parse and pre-process any RTSTRUCT DICOMs in the input.  
    **Default:** `False`

- `-v`, `--verbose`  
    Enable detailed logging and progress messages.

- `-f`, `--force`  
    Overwrite any existing preprocessing or segmentation outputs in the output directory.

- `-ppo`, `--preprocess_only`  
    Only perform preprocessing. No segmentation or postprocessing will occur.

- `-pso`, `--postprocess_only`  
    Only perform postprocessing. Assumes that segmentations already exist.

- `-suv`, `--suv_threshold`  
    Apply an SUV threshold to the lesion segmentation output.  
    **Default:** `0.0`

- `-or`, `--organ_dir`  
    Path to directory containing organ segmentation masks for lesion classification.  
    **Default:** `.../output_dir.parent/organ_segmentations`

---

## Expected Input Structure

This tool supports both DICOM and NIfTI inputs. The expected structure varies slightly depending on the format:

### DICOM Input

The root `input_dir` should contain case folders, each representing a patient + acquisition. Each case folder should contain study-level folders, which in turn contain modality-specific DICOM subfolders:

```bash
input_dir/
├── case1/
│   ├── study/
│   │   ├── CT_001/           ← contains CT DICOM `.dcm` files
│   │   └── PET_001/          ← contains PET DICOM `.dcm` files
├── case2/
│   └── study/
│       ├── CT_series/
│       └── PT_series/
...
```

Each study folder must contain at least one CT and one valid PET series. 
PET series are validated for required DICOM tags needed for SUV conversion (e.g., `CorrectedImage`, `DecayCorrection`, `Units`, etc.).

### NIfTI Input

If the pipeline detects any `.nii.gz` files anywhere under `input_dir`, it assumes the entire input is NIfTI-based and processes the files in either a flattened or per-case format:

#### Option 1: Flat NIfTI Files
```bash
input_dir/
├── patient1_0000.nii.gz  ← CT
├── patient1_0001.nii.gz  ← PET
├── patient2_0000.nii.gz
└── patient2_0001.nii.gz
...
```

#### Option 2: Case Subfolders
```bash
input_dir/
├── patient1/
│   ├── ct.nii.gz
│   └── pet.nii.gz
├── patient2/
│   ├── ct.nii.gz
│   └── pet.nii.gz
...
```

The pipeline recursively scans all `.nii.gz` files and groups them by case using the filename pattern `caseid_0000.nii.gz` (CT) and `caseid_0001.nii.gz` (PET). This aligns with [nnUNet's](https://github.com/MIC-DKFZ/nnUNet) naming convention. 

All files must already be co-registered and in consistent orientation (LPS assumed). Additionally, all PET images already in `nii.gz` format are assumed to be SUV-converted.

If CT and PET images differ in shape, the CT will be automatically resampled to the PET image.

### RTSTRUCT Support (Optional)

If `--include_rtstructs` is enabled and an RTSTRUCT series is present alongside the CT and PET series' within each study folder, the RTSTRUCT `.dcm` file will be converted - using Plastimatch - into individual NIfTI masks, with optional renaming applied.

---

## Inference and Output

An `nnUNetPredictor` is used for inference with the downloaded model weights, with the output predictions being saved to the specified (or default) `output_dir`. 

---

## Post-processing

After segmentation, post-processing is performed to classify lesions and extract biomarkers. This includes:

#### SUV thresholding (optional):
  If an SUV threshold is provided (e.g., `-suv 3.0`), all voxels below this value are removed from the segmentation results. If `overwrite == False`, the non-thresholded predictions are preserved in a backup folder.

#### Organ segmentation generation:
  If not provided, organ segmentations are automatically generated using `TotalSegmentator` and used to classify lesions into anatomical regions.

#### Lesion classification and metrics extraction:
  Each lesion is assigned to an organ or classified as nodal (either above or below the Common Iliac Bifurcation) based on an overlap threshold, and key metrics are extracted at the lesion- and patient-level.

#### Results JSON output:

A `lesion_results.json` (or `lesion_results_suv_thresh_{X}.json` if SUV thresholding is applied) is saved in the output directory. It contains an entry for each case, consisting of:
- `lesions`: Dictionary providing the `TotalSegmentator` site code and name, and volume (in cc) for each segmented lesion.
- `lesion_metrics`: A collated dictionary containing:

    - `site`: Aggregated lesion counts and volumes by specific `TotalSegmentator` site.
    - `region`: Aggregated metrics grouped into higher-level regions (whole body, bone, nodal above/below CIB, visceral, and prostate).
    - `patient`: SUV mean, max, and total metrics.

A final 'All' entry collates the site- and region-level metrics across all provided cases. 

This step runs automatically after inference, unless the output JSON already exists and `--overwrite` is not set.