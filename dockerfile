# GPU-enabled Dockerfile for PSMA Segmentator
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Environment settings
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDNN_CONV_ALGO_SEARCH=DEFAULT \
    CUDNN_WORKSPACE_LIMIT=4096 \
    TORCH_CUDNN_V8_ENABLE_WEIGHT_GRADIENT_MEMORY_EFFICIENT_FUSION=1 \
    PYTORCH_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True \
    NNUNET_PERFORM_EVERYTHING_ON_DEVICE=0 \
    NNUNET_NUM_PREPROCESSING_WORKERS=0 \
    NNUNET_NUM_NIFTI_SAVE_WORKERS=0 \
    NNUNET_NO_GPU_PREPROCESSING=1 \
    NNUNET_FORCE_CPU_STITCHING=1 \
    CUDNN_BENCHMARK=0 \
    CUDNN_DETERMINISTIC=1 \
    VENV_PATH=/opt/venv

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    git xvfb curl tar \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Create virtual environment
RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Upgrade pip for Python 3.11
RUN python -m pip install --upgrade pip setuptools wheel

# Install Plastimatch prebuilt binary from SourceForge
RUN curl -L -o /tmp/plastimatch.tar.bz2 https://sourceforge.net/projects/plastimatch/files/Source/plastimatch-1.9.4.tar.bz2 && \
    tar -xjf /tmp/plastimatch.tar.bz2 -C /usr/local/bin --strip-components=1 && \
    rm /tmp/plastimatch.tar.bz2

# Install PyTorch + torchvision + torchaudio matching bare-metal
RUN python -m pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Working directory
WORKDIR /app

# Copy repo files
COPY . /app

# Install Python dependencies
RUN python -m pip install -e .

# Add runtime check script
RUN echo 'import torch\nprint("PyTorch:", torch.version)\nprint("CUDA:", torch.version.cuda)\nprint("cuDNN:", torch.backends.cudnn.version())\nprint("Device count:", torch.cuda.device_count())' > /app/check_env.py

# Entry point for CLI
ENTRYPOINT ["python", "-m", "psma_segmentator.cli"]
