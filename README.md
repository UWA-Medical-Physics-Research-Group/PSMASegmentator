# PSMASegmentator

**PSMASegmentator** is a tool for the automatic segmentation of PSMA PET/CT images using Deep Learning (DL). 

The segmentation model uses the [nnUNetv2 framework](https://github.com/MIC-DKFZ/nnUNet), along with the DKFZ LesionTracer team's winning [autoPET-III methodology](https://github.com/mic-dkfz/autopet-3-submission). 
The training dataset is across multiple institutions and scanner types, with 597 patients from the [autoPET-III dataset](https://autopet-iii.grand-challenge.org/) and 438 from Western Australian institutions, for a total of 1035 patients.

It supports both DICOM and NIfTI inputs and automatically handles pre-processing, inference, and post-processing. This includes lesion classification and biomarker extraction, utilizing [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) and a complimentary liver metastases classifier model.

---
---

## Installation

It is recommended to use a [Miniconda](https://docs.conda.io/en/latest/miniconda.html) virtual environment. Once you've [installed Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions):

```bash
conda create -n psma_segmentator python=3.11 -y
conda activate psma_segmentator
```

Then, clone the repository. As this repo is currently private, you will need a Personal Access Token (PAT) to clone the repo and use the API. To generate a PAT, click [here](https://github.com/settings/tokens) then select **Generate new token --> Classic** and give it 'repo' scope. 


Once you have your PAT, use it and your GitHub username to clone the repo, as shown below:

```bash
git clone https://<your-username>:<your-PAT>@github.com/UWA-Medical-Physics-Research-Group/PSMASegmentator /path/to/where/you/want/PSMASegmentator
```

Then move to the cloned repo location before installing in editable mode:

```bash
cd path/to/where/you/put/PSMASegmentator
pip install -e .
```

### Installing Plastimatch (required for RTSTRUCT output)

If you want to convert any input RTSTRUCT files to NIfTIs and/or convert the output segmentations to RTSTRUCT format, you'll need `plastimatch`. This can be installed in two main ways:

#### Option 1: Install via apt

```bash
sudo apt install plastimatch
```

#### Option 2 — Install via PyPlastimatch (for any distro without apt plastimatch)

Some distributions (notably Ubuntu 22.04) do not provide a `plastimatch` package via apt.
In these cases, you can install a precompiled `plastimatch` binary using the `PyPlastimatch` installation utility.

1. Install the Python package:

`pip install pyplastimatch`

2. Run the precompiled binary installer using your conda environment’s Python with `sudo` so it can write to `/usr/local/bin`:

```bash
sudo ~/miniconda3/envs/psma_segmentator/bin/python -c \
"from pyplastimatch.utils.install import install_precompiled_binaries; install_precompiled_binaries()"
```

This downloads the correct Ubuntu‑22.04 plastimatch binary and installs it system‑wide.

3. Verify the installation:
```bash
plastimatch --version
which plastimatch
```

You should see:
```bash
plastimatch version 1.9.4-XX
/usr/local/bin/plastimatch
```

---
---

## Usage - CLI

Once installed, you can run segmentations using the command-line interface (CLI):

```bash
python -m psma_segmentator.cli -i INPUT_DIR -pat YOUR_TOKEN [options]
```

### Required Arguments

- `-i`, `--input_dir`  
    Path to input directory containing the images to be segmented. These can be in DICOM or NIfTI format.
#### OR:
- `-i_ct`, `--input_ct`  
    Path to input CT NIfTI file to be segmented.  
#### AND
- `-i_pet`, `--input_pet`  
    Path to input PET NIfTI file to be segmented.  

- `-pat`, `--patient_token`  
    Your patient-specific token for accessing releases from the `PSMASegmentator` GitHub repository (see above).  

### Optional Arguments

- `-o`, `--output_dir`  
    Path to save the output `.nii.gz` segmentation files to. Recommended to specify this when passing direct NIfTIs via `-i_ct` and `-i_pet`.
    **Default:** `.../[input].parent/[input].name_outputs`

- `-w`, `--weights_dir`          
    Path to either existing weights directory, or directory to download weights to.  
    **Default:** `~/.psmasegmentator/[version]`

- `-chkpt`, `--checkpoint_name`     
    Specify the name of the .pth file to use for inference (defaults to `checkpoint_final.pth`).

- `-plans`, `--plans_name`  
    Name of the plans JSON file to use for inference (e.g., if changing patch size). 

- `--version`  
    Specify which release of `PSMASegmentator` to use (e.g., `v1.0.0`).  
    **Default:** Latest available release.

- `-d`, `--device`  
    Choose the device for inference.  
    **Options:** `"cpu"`, `"cuda"` or `"cuda:n"` (where `n` is the GPU index)  
    **Default:** `"cuda"` if available.

- `-rts`, `--rtstruct_processing`  
    If True, will convert any found RTSTRUCTs to NIfTI and convert output NIfTIs  to RTSTRUCTs.  
    **Default:** `False`

- `-ppo`, `--preprocess_only`  
    Only perform preprocessing. No segmentation or postprocessing will occur.

- `-dpp`, `--disable_postprocessing`  
    Disable post-processing of the output files to just do segmentation.

- `-suv`, `--suv_threshold`  
    Apply an SUV threshold to the lesion segmentation output.  
    **Default:** `0.0`

- `-exp_segs`, `--expand_segmentations`
    Expand output segmentations during post-processing.

- `-or`, `--organ_dir`  
    Path to directory containing organ segmentation masks for lesion classification.  
    **Default:** `.../output_dir.parent/organ_segmentations`

- `--fast`  
    Use fast mode for inference. This uses the Fast (lightweight) version of PSMASegmentator, disables Test-Time Augmentation (TTA), and uses the `--fast` flag in TotalSegmentator for faster organ segmentation generation. Note that only the weights for the Fast model are downloaded/accessed from the given GitHub release when this flag is provided.

- `-f`, `--force`  
    Overwrite any existing preprocessing or segmentation outputs in the output directory.

- `-v`, `--verbose`  
    Enable detailed logging and progress messages.

- `--save_log`  
    Save the entire CLI stdout/stderr to a timestamped `.txt` file in the parent directory of `--output_dir` (or parent of `--input_dir`, or `cwd` if neither).

---

## Usage – Docker

The following explains how to use PSMASegmentator via Docker.
You may either:

1. **Pull the published Docker image** from the repository registry, or
2. **Load a pre-built Docker image** provided as a `.tar.gz` file, or
3. **Build the image yourself** directly from the repository.

All approaches result in a Docker image named `psma-segmentator:latest`.

---

### 1. Prerequisites

* [Docker installed](https://docs.docker.com/get-docker/) (version 20.10+ recommended)

* [NVIDIA GPU drivers + NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (required for GPU support)

* **System requirements**

  * **Shared memory (`--shm-size`)**:
    32 GB recommended for large whole-body PET/CT images.

  * **GPU VRAM**:
    At least 12–24 GB recommended depending on patch size

  * Sufficient disk space for input, output, and the Docker image

---

### 2. Pulling the Published Docker Image

The image is published to the GitHub Container Registry (GHCR). If the repository is private, you must log in with a GitHub Personal Access Token (PAT) that has **read:packages** scope.

```bash
docker login ghcr.io -u <github-username> -p <PAT>
docker pull ghcr.io/uwa-medical-physics-research-group/psmasegmentator:latest
docker tag ghcr.io/uwa-medical-physics-research-group/psmasegmentator:latest psma-segmentator:latest
```

To pull a specific release tag:

```bash
docker pull ghcr.io/uwa-medical-physics-research-group/psmasegmentator:<version>
docker tag ghcr.io/uwa-medical-physics-research-group/psmasegmentator:<version> psma-segmentator:latest
```

---

### 3. Using a Pre-Built Docker Image (`.tar.gz`)

If using a Docker image file such as:
```
psma-segmentator_<yyyymmdd>.tar.gz
```

#### Load the image into Docker

```bash
cd /path/to/image
docker load -i psma-segmentator_<yyyymmdd>.tar.gz
```

Verify the image is now available:

```bash
docker images
```

---

### 4. Building the Docker Image Yourself

If opting to build the image locally from the repository:

```bash
cd /path/to/psma-segmentator
DOCKER_BUILDKIT=1 docker build -t psma-segmentator:latest .
```

---

### 5. Prepare Input and Weight Directories

#### Input directory example

```
/home/<user>/data/
```

Contains:

```
CT_0000.nii.gz
PT_0001.nii.gz
```

#### Create weights directories (required for permissions)

```bash
mkdir -p /home/<user>/.psmasegmentator
mkdir -p /home/<user>/.totalsegmentator
```

---

### 6. Running PSMA Segmentator via Docker

Below is the recommended full command:

```bash
docker run --rm --gpus all \
    --user $(id -u):$(id -g) \
    --shm-size=32g \
    -v /home/<user>/<data>:/data \
    -v /home/<user>/.psmasegmentator:/weights \
    -v /home/<user>/.totalsegmentator:/ts_weights \
    -e TOTALSEG_HOME_DIR=/ts_weights \
    psma-segmentator:latest \
    -plans 'plans_reduced_patch.json' \
    -i_ct /data/CT_0000.nii.gz \
    -i_pet /data/PT_0001.nii.gz \
    -o /data/psmasegmentator_outputs \
    -pat <PAT> \
    -w /weights/1.0.0
```

---

#### Explanation of Key Options

#### Docker runtime options

| Option                     | Meaning                                    |
| -------------------------- | ------------------------------------------ |
| `--rm`                     | Remove container after completion          |
| `--gpus all`               | Use all available GPUs                     |
| `--user $(id -u):$(id -g)` | Ensures output files are not owned by root |
| `--shm-size=32g`           | Required for handling large 3D arrays      |
| `-v <host>:<container>`    | Mount directories into the container       |
| `-e TOTALSEG_HOME_DIR`     | Set TotalSegmentator download location     |  

#### Container arguments

| Argument | Purpose                               |
| -------- | ------------------------------------- |
| `-plans` | Network plan JSON file                |
| `-i_ct`  | CT input image                        |
| `-i_pet` | PET input image                       |
| `-i`     | Input directory                       |
| `-o`     | Output directory                      |
| `-pat`   | GitHub Personal Access Token          |
| `-w`     | Model weights folder inside container |

See 'CLI Usage' above for an explanation of the other optional arguments.

---
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

If the pipeline detects any `.nii.gz` files anywhere under `input_dir`, it assumes the entire input is NIfTI-based and processes the files in either a flattened or case-subfolder format:

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

#### Option 3: Direct NIfTI Input
Alternatively, you can pass direct paths to CT and PET NIfTI files using the `-i_ct` and `-i_pet` flags. *In this case, it is recommended to specify an `--output_dir` as well, as the default output directory will be specific to the input file which can bloat the parent directory*.

All files must already be co-registered and in consistent orientation (LPS assumed). Additionally, *all PET images already in `nii.gz` format are assumed to be SUV-converted*.

If CT and PET images differ in shape, the CT will be automatically resampled to the PET image.

### RTSTRUCT Support (Optional)

If `-rts`/`--rtstruct_processing` is enabled and an RTSTRUCT series is present alongside the CT and PET series' within each study folder, the RTSTRUCT `.dcm` file will be converted - using `Plastimatch` - into individual NIfTI masks, with optional renaming applied. Additionally, output NIfTI masks will be converted into an RTSTRUCT series and saved in a dedicated '_rtstructs' directory alongside the output directory.

---
---

## Inference and Output Segmentations

An `nnUNetPredictor` is used for inference with the downloaded model weights, with the output predictions being saved to the specified (or default) `--output_dir`. 

---
---

## Post-processing

After segmentation, post-processing is performed to classify lesions and extract biomarkers. This includes:

#### SUV thresholding (optional):
  If an SUV threshold is provided (e.g., `-suv 3.0`), all voxels below this value are removed from the segmentation results. If `overwrite == False`, the non-thresholded predictions are preserved in a backup folder.

#### Segmentation expansion (optional):
  Expand initial output segmentations. This uses CCA Fast Marching with an adaptive SUV threshold, along with organ-aware constraints, a maximum volume expansion factor, and watershed filtering, to expand the segmentation model's outputs. It's been shown to markedly improve voxel-level sensitivity, with a minor trade-off in voxel-level precision.

#### Organ segmentation generation:
  If not provided, organ segmentations are automatically generated using `TotalSegmentator` and used to classify lesions into anatomical regions.

#### Lesion classification and metrics extraction:
  Each lesion is assigned to an organ or classified as nodal (either above or below the Common Iliac Bifurcation) based on an overlap threshold, and key metrics are extracted at the lesion- and patient-level.

#### Liver disease classification:
  A complimentary binary classifier model is present to detect the presence of liver metastases, a significant negative prognosticator.

---

## Output Summary Files

The following output files are saved in the specified output directory or, by default, the `[input_folder]_lesion_classification` sub-folder.

### Results JSON:

A `lesion_results.json` (or `lesion_results_suv_thresh_{X}.json` if SUV thresholding is applied). It contains an entry for each case, consisting of:
- `lesions`: Dictionary providing the `TotalSegmentator` site code and name, and volume (in cc) for each segmented lesion.
- `lesion_metrics`: A collated dictionary containing:

    - `site`: Aggregated lesion counts and volumes by specific `TotalSegmentator` site.
    - `region`: Aggregated metrics grouped into higher-level regions (whole body, bone, nodal above/below CIB, visceral, and prostate).
    - `patient`: SUVmean, SUVmax, Total Lesion Uptake (TLU), Total Lesion Quotient (TLQ), and whether liver metastases have been detected.

A final 'All' entry collates the site- and region-level metrics across all provided cases. 

This step runs automatically after inference, unless the output JSON already exists and `--overwrite` is not set.

### Results CSV:

A `biomarker_info.csv` file. Each row corresponds to a `Case`, with column headings of:

- Number of lesions
- PSMA Total Tumour Volume (TTV)
- Tumour SUVmean
- Tumour SUVmax
- Total Lesion Uptake (TLU)
- Total Lesion Quotient (TLQ)
- Bone metastases present (True/False)
- Visceral metastases present (True/False)
- Liver metastases present (True/False)

---