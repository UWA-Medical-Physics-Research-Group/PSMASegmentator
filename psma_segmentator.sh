#!/usr/bin/env bash
set -e

IMAGE_NAME="psma-segmentator:latest"
DOCKERFILE_PATH="./Dockerfile"

#############################################
# 1. Build image if missing
#############################################
if [[ "$(docker images -q $IMAGE_NAME 2>/dev/null)" == "" ]]; then
    echo "[INFO] Docker image not found. Building $IMAGE_NAME ..."
    docker build -t $IMAGE_NAME -f $DOCKERFILE_PATH .
fi

#############################################
# 2. Parse arguments: detect --weights and --input
#############################################
USER_WEIGHTS_DIR=""
USER_INPUT_DIR=""
PARSED_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --weights|-w)
            USER_WEIGHTS_DIR="$2"
            shift 2
            ;;
        --input|-i)
            USER_INPUT_DIR="$2"
            shift 2
            ;;
        *)
            PARSED_ARGS+=("$1")
            shift
            ;;
    esac
done

set -- "${PARSED_ARGS[@]}"

#############################################
# 3. Resolve and validate weights directory
#############################################
if [[ -n "$USER_WEIGHTS_DIR" ]]; then
    WEIGHTS_DIR="$USER_WEIGHTS_DIR"
else
    WEIGHTS_DIR="$HOME/.psmasegmentator/weights"
fi

mkdir -p "$WEIGHTS_DIR"

WEIGHTS_MOUNT=(-v "$WEIGHTS_DIR":/weights)

echo "[INFO] Using weights dir: $WEIGHTS_DIR"

#############################################
# 4. Resolve and validate input directory
#############################################
if [[ -z "$USER_INPUT_DIR" ]]; then
    echo "ERROR: You must specify --input <path>"
    exit 1
fi

if [[ ! -d "$USER_INPUT_DIR" ]]; then
    echo "ERROR: Input directory not found: $USER_INPUT_DIR"
    exit 1
fi

# Resolve absolute path
INPUT_DIR_REAL=$(realpath "$USER_INPUT_DIR")

# Mount the parent directory so all siblings are visible
INPUT_PARENT=$(dirname "$INPUT_DIR_REAL")

INPUT_MOUNT=(-v "$INPUT_PARENT":/data)

echo "[INFO] Mounting input data root: $INPUT_PARENT"
echo "[INFO] In-container input = /data/$(basename "$INPUT_DIR_REAL")"

#############################################
# 5. Shared memory + env flags
#############################################
SHM_SIZE="32g"
ENV_FLAGS=(-e "IS_DOCKER=1")

#############################################
# 6. Run container with mounts + passthrough args
#############################################
docker run --rm \
    --gpus all \
    --shm-size="$SHM_SIZE" \
    "${INPUT_MOUNT[@]}" \
    "${WEIGHTS_MOUNT[@]}" \
    "${ENV_FLAGS[@]}" \
    $IMAGE_NAME \
    "$@"
