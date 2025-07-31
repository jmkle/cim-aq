#!/bin/bash

# Get the directory of the script and the repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# This script handles RL-based quantization search:
# 1. Generate hardware latency lookup table
# 2. Run RL-based search for mixed precision using INT8 model as starting point

# Usage: bash run_rl_quantize.sh [quant_model] [dataset] [dataset_root] [max_accuracy_drop] [min_bit] [max_bit] [train_episodes] [search_finetune_epochs] [force_first_last_layer] [consider_cell_resolution] [output_suffix] [finetune_lr] [uniform_model_file] [wandb_enable] [wandb_project] [gpu_id]
# Example: bash run_rl_quantize.sh qvgg16 imagenet100 /path/to/dataset 1.0 2 8 600 3 true false accdrop1.0bit28 0.001 /path/to/uniform/model.pth.tar false cim-aq-quantization 1

# Default values
QUANT_MODEL=${1:-"qvgg16"}                          # Quantized model architecture
DATASET=${2:-"imagenet100"}                         # Dataset name
DATASET_ROOT=${3:-"${REPO_ROOT}/data/imagenet100"}  # Path to dataset
MAX_ACCURACY_DROP=${4:-"1.0"}                       # Maximum allowed accuracy drop in percentage points
MIN_BIT=${5:-"1"}                                   # Minimum bit-width
MAX_BIT=${6:-"8"}                                   # Maximum bit-width
TRAIN_EPISODES=${7:-"600"}                          # Number of training episodes for RL search
SEARCH_FINETUNE_EPOCHS=${8:-"3"}                    # Number of epochs for fine-tuning during search
FORCE_FIRST_LAST_LAYER=${9:-"true"}                 # Force first and last layers to high precision
CONSIDER_CELL_RESOLUTION=${10:-"false"}             # Whether to consider cell resolution for possible weight bit widths
OUTPUT_SUFFIX=${11:-"accdrop${MAX_ACCURACY_DROP}bit${MIN_BIT}${MAX_BIT}"}
FINETUNE_LR=${12:-"0.001"}                          # Learning rate for fine-tuning during RL search
UNIFORM_MODEL_FILE="${13:-"${REPO_ROOT}/checkpoints/${QUANT_MODEL}_per-tensor_uniform_8bit/model_best.pth.tar"}"  # Path to INT8 model file
WANDB_ENABLE=${14:-"false"}                         # Enable W&B logging
WANDB_PROJECT=${15:-"cim-aq-quantization"}          # W&B project name
GPU_ID=${16:-"1"}                                   # GPU ID(s) for CUDA_VISIBLE_DEVICES

BASE_MODEL_NAME=${QUANT_MODEL/q/}

# Convert string to arg for Python scripts
if [[ "$FORCE_FIRST_LAST_LAYER" == "true" || "$FORCE_FIRST_LAST_LAYER" == "True" || "$FORCE_FIRST_LAST_LAYER" == "1" ]]; then
  FORCE_FIRST_LAST_CLI_ARG="--force_first_last_layer"
else
  FORCE_FIRST_LAST_CLI_ARG="--no-force_first_last_layer"
fi

if [[ "$CONSIDER_CELL_RESOLUTION" == "true" || "$CONSIDER_CELL_RESOLUTION" == "True" || "$CONSIDER_CELL_RESOLUTION" == "1" ]]; then
  CONSIDER_CELL_RESOLUTION_CLI_ARG="--consider_cell_resolution"
else
  CONSIDER_CELL_RESOLUTION_CLI_ARG="--no-consider_cell_resolution"
fi

if [[ "$WANDB_ENABLE" == "true" || "$WANDB_ENABLE" == "True" || "$WANDB_ENABLE" == "1" ]]; then
  WANDB_CLI_ARG="--wandb_enable"
else
  WANDB_CLI_ARG=""
fi

echo "========================================================="
echo "Starting RL-based quantization search for $QUANT_MODEL"
echo "Dataset: $DATASET at $DATASET_ROOT"
echo "Maximum accuracy drop: $MAX_ACCURACY_DROP%"
echo "Bit-width range: $MIN_BIT-$MAX_BIT bits"
echo "RL training episodes: $TRAIN_EPISODES"
echo "Search fine-tuning epochs: $SEARCH_FINETUNE_EPOCHS"
echo "Force first/last layer high precision: $FORCE_FIRST_LAST_LAYER"
echo "Consider cell resolution: $CONSIDER_CELL_RESOLUTION"
echo "Output suffix: $OUTPUT_SUFFIX"
echo "Finetune learning rate: $FINETUNE_LR"
echo "GPU ID: $GPU_ID"
echo "W&B logging: $WANDB_ENABLE"
if [ "$WANDB_ENABLE" = "true" ]; then
  echo "W&B project: $WANDB_PROJECT"
fi
echo "Repository root: $REPO_ROOT"
echo "========================================================="

# Check if INT8 model exists
if [ ! -f "$UNIFORM_MODEL_FILE" ]; then
  echo "Error: INT8 fine-tuned model not found at $UNIFORM_MODEL_FILE"
  echo "Please run INT8 pretraining first."
  exit 1
fi

# Create symlink to ensure RL-based search uses our 8-bit model
echo ""
echo "Creating symlink to 8-bit model for RL-based search..."
mkdir -p "${REPO_ROOT}/pretrained/imagenet"
ln -sf "$UNIFORM_MODEL_FILE" "${REPO_ROOT}/pretrained/imagenet/${QUANT_MODEL}.pth.tar"

# Step 1: Generate hardware latency lookup table
echo ""
echo "Step 1/2: Generating hardware latency lookup table..."
# Make the hardware config script executable
chmod +x "${SCRIPT_DIR}/run_hardware_config.sh"

# Determine layer dimensions file based on model name
LAYER_DIMS_FILE="${BASE_MODEL_NAME}_layer_dimensions.yaml"

bash "${SCRIPT_DIR}/run_hardware_config.sh" $QUANT_MODEL $MAX_BIT $LAYER_DIMS_FILE "hardware_config.yaml"

# Step 2: Run RL-based quantization search
echo ""
echo "Step 2/2: Running RL-based quantization search from 8-bit model..."
echo "Maximum accuracy drop allowed from 8-bit model: $MAX_ACCURACY_DROP%"

# rl_quantize.py always uses pretrained/imagenet/${MODEL}.pth.tar as the starting point
python "${REPO_ROOT}/rl_quantize.py" \
  --arch $QUANT_MODEL \
  --dataset $DATASET \
  --dataset_root $DATASET_ROOT \
  --suffix "${OUTPUT_SUFFIX}_from_8bit" \
  --preserve_ratio 1.0 \
  --float_bit 8 \
  --max_bit $MAX_BIT \
  --min_bit $MIN_BIT \
  --n_worker 32 \
  --data_bsize 128 \
  --train_size 20000 \
  --val_size 10000 \
  --acc_drop $MAX_ACCURACY_DROP \
  --acc_constraint \
  $CONSIDER_CELL_RESOLUTION_CLI_ARG \
  $FORCE_FIRST_LAST_CLI_ARG \
  --finetune_epoch $SEARCH_FINETUNE_EPOCHS \
  --finetune_lr $FINETUNE_LR \
  --finetune_gamma 0.9 \
  --warmup 30 \
  --init_delta 0.6 \
  --delta_decay 0.997 \
  --train_episode $TRAIN_EPISODES \
  --lr_a 5e-5 \
  --lr_c 1e-3 \
  --bsize 64 \
  --rmsize 128 \
  --gpu_id $GPU_ID \
  $WANDB_CLI_ARG \
  --wandb_project "$WANDB_PROJECT"

# Check if best strategy file exists
STRATEGY_FILE="${REPO_ROOT}/save/${QUANT_MODEL}_${DATASET}_${OUTPUT_SUFFIX}_from_8bit/best_policy.npy"
if [ ! -f "$STRATEGY_FILE" ]; then
  echo "Error: Strategy file not found. Search may have failed."
  exit 1
fi

echo ""
echo "========================================================="
echo "RL-based quantization search complete!"
echo "Mixed precision strategy: $STRATEGY_FILE"
echo "========================================================="
