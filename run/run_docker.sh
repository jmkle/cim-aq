#!/bin/bash
##############################################################################
# Copyright (C) 2025 Joel Klein                                              #
# All Rights Reserved                                                        #
#                                                                            #
# This work is licensed under the terms described in the LICENSE file        #
# found in the root directory of this source tree.                           #
##############################################################################

# CIM-AQ Docker Runner - Simplified container management

set -e

# Get directory of script
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="${SCRIPT_DIR}/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
PROJECT_DIR=${SCRIPT_DIR}/..

# Defaults
IMAGE="ghcr.io/jmkle/cim-aq:latest"
GPU_FLAG="--gpus all"
PULL_IMAGE="auto"

# Usage
usage() {
  cat << EOF
Usage: $0 [OPTIONS] [CONFIG_FILE]

OPTIONS:
    -h, --help          Show help
    -t, --test          Run test workflow
    --tag TAG           Docker image tag (default: latest)
    --image IMAGE       Full Docker image name (overrides default)
    --pull WHEN         When to pull image:
                          auto    - Pull if tag is 'latest', 'main', or 'pr-*' (default)
                          always  - Always pull before running
                          never   - Never pull, use local image only
    --gpu SPEC          GPU specification:
                          all     - Use all GPUs (default)
                          none    - No GPU support
                          0,1,2   - Specific GPU IDs
                          device=0 - Single GPU device
    --data PATH         Path to data directory (must contain subdirectories for the datasets)

EXAMPLES:
    $0                                              # Interactive bash
    $0 --test                                       # Test workflow
    $0 config.yaml                                  # Custom workflow
    $0 --tag main --test                            # Use main branch
    $0 --image ghcr.io/jmkle/cim-aq:pr-123 --test   # Use specific image
    $0 --pull always --test                         # Always pull latest before testing
    $0 --pull never --test                          # Use local image only
    $0 --gpu 0,1 --test                             # Use GPUs 0 and 1
    $0 --gpu none --test                            # No GPU support
    $0 --data /path/to/data config.yaml             # Use custom data path with config
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    -t|--test) TEST=1; shift ;;
    --tag) IMAGE="${IMAGE%:*}:$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --pull)
      case "$2" in
        auto|always|never) PULL_IMAGE="$2" ;;
        *) echo "Error: Invalid pull policy '$2'. Must be 'auto', 'always', or 'never'"; usage; exit 1 ;;
      esac
      shift 2 ;;
    --gpu)
      case "$2" in
        all) GPU_FLAG="--gpus all" ;;
        none) GPU_FLAG="" ;;
        *) GPU_FLAG="--gpus \"device=$2\"" ;;
      esac
      shift 2 ;;
    --data) DATA_PATH="$2"; shift 2 ;;
    -*) echo "Unknown option: $1"; usage; exit 1 ;;
    *) CONFIG="$1"; shift ;;
  esac
done

# Check Docker
if ! docker info &>/dev/null; then
  echo "Error: Docker not available"
  exit 1
fi

# Function to test if filesystem supports SELinux extended attributes
supports_selinux_labeling() {
  local test_dir="$1"
  local test_file

  # Create a temporary test file
  test_file=$(mktemp --tmpdir="$test_dir" .selinux_test.XXXXXX 2>/dev/null) || return 1

  # Try to set a SELinux attribute
  if command -v setfattr >/dev/null 2>&1; then
    if setfattr -n security.selinux -v "system_u:object_r:container_file_t:s0" "$test_file" 2>/dev/null; then
      rm -f "$test_file" 2>/dev/null
      return 0
    fi
  fi

  rm -f "$test_file" 2>/dev/null
  return 1
}

# Detect SELinux labeling support
SELINUX_SUFFIX=""
PRIVILEGED_FLAG=""
if supports_selinux_labeling "$PROJECT_DIR"; then
  echo "Filesystem supports SELinux labeling - using :Z suffix"
  SELINUX_SUFFIX=":Z"
else
  echo "Filesystem does not support SELinux labeling - skipping :Z suffix"
  echo "  (This is normal for network filesystems like NFS, CIFS, etc.)"

  # Only use --privileged for podman, which seems to need it on network filesystems
  if [[ "$(docker --version)" == *"podman"* ]]; then
    echo "  Using --privileged flag for podman on network filesystem"
    PRIVILEGED_FLAG="--privileged"
  fi
fi

# Setup volumes using project root
VOLUMES="-v $PROJECT_DIR/checkpoints:/workspace/checkpoints$SELINUX_SUFFIX -v $PROJECT_DIR/save:/workspace/save$SELINUX_SUFFIX"
if [[ -n "$DATA_PATH" ]]; then
  VOLUMES="$VOLUMES -v $DATA_PATH:/workspace/data$SELINUX_SUFFIX"
else
  VOLUMES="$VOLUMES -v $PROJECT_DIR/data:/workspace/data$SELINUX_SUFFIX"
fi
if [[ -n "$CONFIG" ]]; then
  # Handle both absolute and relative config paths
  if [[ "$CONFIG" = /* ]]; then
    CONFIG_ABS="$CONFIG"
  else
    CONFIG_ABS="$(realpath "$CONFIG")"
  fi
  VOLUMES="$VOLUMES -v $CONFIG_ABS:/workspace/$(basename "$CONFIG")$SELINUX_SUFFIX"
fi

# Create directories in project root
mkdir -p "$PROJECT_DIR/checkpoints" "$PROJECT_DIR/save" "$PROJECT_DIR/data"

# Determine if we should pull the image
should_pull=false
case "$PULL_IMAGE" in
  "always")
    should_pull=true
    echo "Pull policy: always - will pull $IMAGE"
    ;;
  "never")
    should_pull=false
    echo "Pull policy: never - using local image $IMAGE"
    ;;
  "auto")
    # Auto pull for 'latest', 'main', or PR tags (which are typically ephemeral)
    if [[ "$IMAGE" == *":latest" ]] || [[ "$IMAGE" == *":main" ]] || [[ "$IMAGE" == *":pr-"* ]]; then
      should_pull=true
      echo "Pull policy: auto - will pull $IMAGE (detected mutable tag)"
    else
      should_pull=false
      echo "Pull policy: auto - using local image $IMAGE (detected immutable tag)"
    fi
    ;;
esac

# Pull image if needed
if [ "$should_pull" = true ]; then
  echo "Pulling Docker image: $IMAGE"
  if ! docker pull "$IMAGE"; then
    echo "❌ Failed to pull image $IMAGE"
    echo "   Continuing with local image if available..."
  else
    echo "✅ Successfully pulled $IMAGE"
  fi
fi

DOCKER_FLAGS=""

if [[ "$(docker --version)" == *"podman"* ]]; then
  echo "Using podman"
  if podman run --help | grep -q -- "--userns=keep-id"; then
    DOCKER_FLAGS="--userns=keep-id"
  else
    DOCKER_FLAGS="--userns=host"
  fi
else
  echo "Using docker"
  # For Docker, try to avoid --privileged unless really necessary
  if [[ -n "$PRIVILEGED_FLAG" ]]; then
    echo "Skipping --user flag when using --privileged to avoid permission conflicts"
  else
    DOCKER_FLAGS="--user $(id -u):$(id -g)"
  fi
fi

# Determine if we're in a CI environment or have TTY
TTY_FLAGS=""
if [[ -t 0 ]] && [[ -t 1 ]] && [[ -z "$CI" ]] && [[ -z "$GITHUB_ACTIONS" ]]; then
  # We have a TTY and not in CI - use interactive mode
  TTY_FLAGS="-it"
else
  # No TTY or in CI environment - non-interactive mode
  TTY_FLAGS=""
fi

if [[ -n "$TEST" ]]; then
  echo "Running test workflow..."
  docker run $TTY_FLAGS --rm $GPU_FLAG $PRIVILEGED_FLAG $DOCKER_FLAGS $VOLUMES $IMAGE bash -c "pip install pillow && ./utils/create_test_data.sh && bash run/run_full_workflow.sh run/configs/test_config.yaml"
elif [[ -n "$CONFIG" ]]; then
  echo "Running workflow with $(basename "$CONFIG")..."
  docker run $TTY_FLAGS --rm $GPU_FLAG $PRIVILEGED_FLAG $DOCKER_FLAGS $VOLUMES $IMAGE bash -c "bash run/run_full_workflow.sh $(basename "$CONFIG")"
else
  echo "Starting interactive session..."
  # For interactive sessions, we always want TTY
  docker run -it --rm $GPU_FLAG $PRIVILEGED_FLAG $DOCKER_FLAGS $VOLUMES $IMAGE bash
fi
