#!/bin/bash
##############################################################################
# Copyright (C) 2025 Joel Klein                                              #
# All Rights Reserved                                                        #
#                                                                            #
# This work is licensed under the terms described in the LICENSE file        #
# found in the root directory of this source tree.                           #
##############################################################################

# Exit on any error, undefined variable, or pipe failure
set -euo pipefail

# Get the directory of the script and the repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# This script handles FP32 model pretraining:
# 1. Download pretrained weights from torchvision
# 2. Create a dummy strategy file for FP32 model evaluation
# 3. Fine-tune the FP32 model to adjust the last layer for new classes
# 4. Evaluate the pretrained model to get baseline accuracy

# Usage: bash run_fp32_pretraining.sh [fp32_model] [dataset] [dataset_root] [fp32_finetune_epochs] [dataset_suffix] [learning_rate] [wandb_enable] [wandb_project] [gpu_id] [batch_size] [num_workers] [amp_enable]
# Example: bash run_fp32_pretraining.sh custom_vgg16 imagenet100 /path/to/dataset 5 imagenet100 0.0005 false cim-aq-quantization 1 256 32 true

# Default values
FP32_MODEL=${1:-"custom_vgg16"}                     # Full precision model architecture
DATASET=${2:-"imagenet100"}                         # Dataset name
DATASET_ROOT=${3:-"${REPO_ROOT}/data/imagenet100"}  # Path to dataset
FP32_FINETUNE_EPOCHS=${4:-"5"}                      # Number of epochs for FP32 model fine-tuning
DATASET_SUFFIX=${5:-""}                             # Optional suffix for output directory naming
LEARNING_RATE=${6:-"0.0005"}                        # Learning rate for FP32 model fine-tuning
WANDB_ENABLE=${7:-"false"}                          # Enable W&B logging
WANDB_PROJECT=${8:-"cim-aq-quantization"}           # W&B project name
GPU_ID=${9:-"1"}                                    # GPU ID(s) for CUDA_VISIBLE_DEVICES
BATCH_SIZE=${10:-"256"}                             # Batch size for training
NUM_WORKERS=${11:-"32"}                             # Number of DataLoader workers
AMP_ENABLE=${12:-"true"}                            # Enable Automatic Mixed Precision

BASE_MODEL_NAME=${FP32_MODEL/custom_/}

# Function to automatically detect number of classes from dataset directory
detect_num_classes() {
  local dataset_path="$1"
  local train_dir="${dataset_path}/train"

  if [ -d "$train_dir" ]; then
    # Count subdirectories in train folder (each subdirectory represents a class)
    local class_count=$(find "$train_dir" -maxdepth 1 -type d | wc -l)
    # Subtract 1 because find includes the parent directory itself
    echo $((class_count - 1))
  else
    echo "Error: Cannot detect classes - train directory not found at $train_dir" >&2
    return 1
  fi
}

# Determine the number of classes based on the dataset
if [ "$DATASET" = "imagenet100" ]; then
  NUM_CLASSES=100
elif [ "$DATASET" = "imagenet" ]; then
  NUM_CLASSES=1000
else
  echo "Unknown dataset: $DATASET - attempting automatic detection..."
  DETECTED_CLASSES=$(detect_num_classes "$DATASET_ROOT")
  if [ $? -eq 0 ] && [ "$DETECTED_CLASSES" -gt 0 ] && [ "$DETECTED_CLASSES" -le 10000 ]; then
    NUM_CLASSES=$DETECTED_CLASSES
    echo "Automatically detected $NUM_CLASSES classes from dataset structure"
  else
    echo "Warning: Could not automatically detect classes or invalid count ($DETECTED_CLASSES)"
    echo "Defaulting to 1000 classes"
    NUM_CLASSES=1000
  fi
fi

# Convert W&B enable string to CLI argument
if [[ "$WANDB_ENABLE" == "true" || "$WANDB_ENABLE" == "True" || "$WANDB_ENABLE" == "1" ]]; then
  WANDB_CLI_ARG="--wandb_enable"
else
  WANDB_CLI_ARG=""
fi

# Convert AMP enable string to CLI argument
if [[ "$AMP_ENABLE" == "true" || "$AMP_ENABLE" == "True" || "$AMP_ENABLE" == "1" ]]; then
  AMP_CLI_ARG="--amp"
else
  AMP_CLI_ARG=""
fi

echo "========================================================="
echo "Starting FP32 model pretraining for $FP32_MODEL"
echo "Dataset: $DATASET at $DATASET_ROOT"
echo "FP32 fine-tune epochs: $FP32_FINETUNE_EPOCHS"
echo "Number of classes: $NUM_CLASSES"
echo "Learning rate: $LEARNING_RATE"
echo "GPU ID: $GPU_ID"
echo "AMP (Automatic Mixed Precision): $AMP_ENABLE"
echo "W&B logging: $WANDB_ENABLE"
if [ "$WANDB_ENABLE" = "true" ]; then
  echo "W&B project: $WANDB_PROJECT"
fi
echo "Repository root: $REPO_ROOT"
echo "========================================================="

# Step 1: Download pretrained weights from torchvision
echo "Step 1/4: Downloading pretrained ${BASE_MODEL_NAME} weights from torchvision..."
mkdir -p "${REPO_ROOT}/pretrained/imagenet"
python "${REPO_ROOT}/lib/utils/get_model_weights.py" --model_name ${BASE_MODEL_NAME} --num_classes $NUM_CLASSES --output_dir "${REPO_ROOT}/pretrained/imagenet"

# Check if the command succeeded
if [ $? -ne 0 ]; then
  echo "Error: Failed to run model weights download script."
  exit 1
fi

# Check if the weights were downloaded successfully
TORCHVISION_MODEL_FILE="${REPO_ROOT}/pretrained/imagenet/${BASE_MODEL_NAME}_${NUM_CLASSES}classes.pth.tar"
if [ ! -f "$TORCHVISION_MODEL_FILE" ]; then
  echo "Error: Failed to download pretrained ${BASE_MODEL_NAME} weights."
  exit 1
fi

# Create symbolic link to use these weights for FP32 model
echo "Creating symbolic link for FP32 model..."
ln -sf "$TORCHVISION_MODEL_FILE" "${REPO_ROOT}/pretrained/imagenet/${FP32_MODEL}.pth.tar"
echo "Pretrained weights saved to $TORCHVISION_MODEL_FILE"

# Step 2: Create a dummy strategy file for FP32 model evaluation
echo ""
echo "Step 2/4: Creating dummy strategy file for FP32 model evaluation..."
mkdir -p "${REPO_ROOT}/save/uniform_strategies"
python "${REPO_ROOT}/utils/create_uniform_strategy.py" --arch $FP32_MODEL --w_bit 8 --a_bit 8 --force_first_last_layer --output "${REPO_ROOT}/save/uniform_strategies"

# Check if the command succeeded
if [ $? -ne 0 ]; then
  echo "Error: Failed to create dummy strategy file."
  exit 1
fi

FP32_STRATEGY_FILE="${REPO_ROOT}/save/uniform_strategies/${FP32_MODEL}_w8a8.npy"

if [ ! -f "$FP32_STRATEGY_FILE" ]; then
  echo "Error: Failed to generate dummy strategy file for FP32 model."
  exit 1
fi
echo "Created dummy strategy file: $FP32_STRATEGY_FILE"

# Step 3: Fine-tune the FP32 model
echo ""
echo "Step 3/4: Fine-tuning the FP32 model for ${FP32_FINETUNE_EPOCHS} epochs..."
if [ -n "$DATASET_SUFFIX" ]; then
  FP32_MODEL_DIR="${REPO_ROOT}/checkpoints/${FP32_MODEL}_pretrained_${DATASET_SUFFIX}"
else
  FP32_MODEL_DIR="${REPO_ROOT}/checkpoints/${FP32_MODEL}_pretrained"
fi
mkdir -p $FP32_MODEL_DIR

# Run fine-tuning with the FP32 model
python "${REPO_ROOT}/finetune.py" \
  -a $FP32_MODEL \
  -d $DATASET_ROOT \
  --data_name $DATASET \
  --epochs $FP32_FINETUNE_EPOCHS \
  --lr $LEARNING_RATE \
  --lr_type cos \
  --wd 0.0001 \
  --batch_size $BATCH_SIZE \
  --workers $NUM_WORKERS \
  --pretrained \
  --checkpoint $FP32_MODEL_DIR \
  --strategy_file $FP32_STRATEGY_FILE \
  $AMP_CLI_ARG \
  --gpu_id $GPU_ID \
  $WANDB_CLI_ARG \
  --wandb_project "$WANDB_PROJECT"

# Check if training succeeded
if [ $? -ne 0 ]; then
  echo "Error: FP32 fine-tuning failed."
  exit 1
fi

# Check if FP32 finetuned model exists
FP32_MODEL_FILE="${FP32_MODEL_DIR}/model_best.pth.tar"
if [ ! -f "$FP32_MODEL_FILE" ]; then
  echo "Error: FP32 fine-tuned model not found. Fine-tuning may have failed."
  exit 1
fi

# Step 4: Evaluate the pretrained model to get baseline accuracy
echo ""
echo "Step 4/4: Evaluating FP32 model to get baseline accuracy..."

# Run the evaluation with non-quantized model
# Use /dev/tty if available (normal terminal), otherwise just capture output
if [ bash -c ": >/dev/tty" >/dev/null 2>/dev/null ]; then
  # `/dev/tty` available - show live output and capture
  EVAL_OUTPUT=$(python "${REPO_ROOT}/finetune.py" \
      -a $FP32_MODEL \
      -d $DATASET_ROOT \
      --data_name $DATASET \
      --evaluate \
      --batch_size $BATCH_SIZE \
      --workers $NUM_WORKERS \
      --strategy_file $FP32_STRATEGY_FILE \
      --gpu_id $GPU_ID \
    --resume $FP32_MODEL_FILE 2>&1 | tee /dev/tty)
else
  # No interactive terminal (CI/batch) - capture output and show afterward
  EVAL_OUTPUT=$(python "${REPO_ROOT}/finetune.py" \
      -a $FP32_MODEL \
      -d $DATASET_ROOT \
      --data_name $DATASET \
      --evaluate \
      --batch_size $BATCH_SIZE \
      --workers $NUM_WORKERS \
      --strategy_file $FP32_STRATEGY_FILE \
      --gpu_id $GPU_ID \
    --resume $FP32_MODEL_FILE 2>&1)

  # Show the captured output
  echo "=== FP32 Model Evaluation Output ==="
  echo "$EVAL_OUTPUT"
  echo "===================================="
fi

# Check if evaluation succeeded
if [ $? -ne 0 ]; then
  echo "Error: FP32 model evaluation failed."
  exit 1
fi

# Store the baseline accuracy by parsing the output
BASELINE_ACCURACY=$(echo "$EVAL_OUTPUT" | grep -oP "Test Acc:\s*\K[0-9\.]+")
BASELINE_ACCURACY5=$(echo "$EVAL_OUTPUT" | grep -oP "Test Acc5:\s*\K[0-9\.]+")

# Validate that extraction worked
if [ -z "$BASELINE_ACCURACY" ] || [ -z "$BASELINE_ACCURACY5" ]; then
  echo "Warning: Failed to extract accuracy from evaluation output" >&2
  echo "Evaluation output (last 10 lines):" >&2
  echo "$EVAL_OUTPUT" | tail -10 >&2
  BASELINE_ACCURACY="Unknown"
  BASELINE_ACCURACY5="Unknown"
fi

echo "Baseline accuracy with fine-tuned FP32 model: ${BASELINE_ACCURACY}% (Top-5: ${BASELINE_ACCURACY5}%)"

echo ""
echo "========================================================="
echo "FP32 pretraining complete!"
echo "FP32 fine-tuned model: $FP32_MODEL_FILE"
echo "Baseline accuracy: ${BASELINE_ACCURACY}% (Top-5: ${BASELINE_ACCURACY5}%)"
echo "========================================================="
