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

# This script handles INT8 model pretraining:
# 1. Generate uniform 8-bit quantization strategy
# 2. Fine-tune the model with uniform 8-bit quantization using FP32 model as base
# 3. Evaluate the 8-bit quantized model

# Usage: bash run_int8_pretraining.sh [quant_model] [fp32_model] [dataset] [dataset_root] [uniform_8bit_epochs] [force_first_last_layer] [dataset_suffix] [learning_rate] [wandb_enable] [wandb_project] [gpu_id] [batch_size] [num_workers]
# Example: bash run_int8_pretraining.sh qvgg16 custom_vgg16 imagenet100 /path/to/dataset 10 true imagenet100 0.001 false cim-aq-quantization 1 256 32

# Default values
QUANT_MODEL=${1:-"qvgg16"}                          # Quantized model architecture
FP32_MODEL=${2:-"custom_vgg16"}                     # Full precision model architecture
DATASET=${3:-"imagenet100"}                         # Dataset name
DATASET_ROOT=${4:-"${REPO_ROOT}/data/imagenet100"}  # Path to dataset
UNIFORM_8BIT_EPOCHS=${5:-"10"}                      # Number of epochs for 8-bit pre-fine-tuning
FORCE_FIRST_LAST_LAYER=${6:-"true"}                 # Force first and last layers to high precision
DATASET_SUFFIX=${7:-""}                             # Optional suffix for output directory naming
LEARNING_RATE=${8:-"0.001"}                         # Learning rate for INT8 model pre-fine-tuning
WANDB_ENABLE=${9:-"false"}                          # Enable W&B logging
WANDB_PROJECT=${10:-"cim-aq-quantization"}          # W&B project name
GPU_ID=${11:-"1"}                                   # GPU ID(s) for CUDA_VISIBLE_DEVICES
BATCH_SIZE=${12:-"256"}                             # Batch size for training
NUM_WORKERS=${13:-"32"}                             # Number of DataLoader workers

# Convert string to boolean for Python scripts
if [[ "$FORCE_FIRST_LAST_LAYER" == "true" || "$FORCE_FIRST_LAST_LAYER" == "True" || "$FORCE_FIRST_LAST_LAYER" == "1" ]]; then
  FORCE_FIRST_LAST_CLI_ARG="--force_first_last_layer"
else
  FORCE_FIRST_LAST_CLI_ARG="--no-force_first_last_layer"
fi

if [[ "$WANDB_ENABLE" == "true" || "$WANDB_ENABLE" == "True" || "$WANDB_ENABLE" == "1" ]]; then
  WANDB_CLI_ARG="--wandb_enable"
else
  WANDB_CLI_ARG=""
fi

echo "========================================================="
echo "Starting INT8 model pretraining for $QUANT_MODEL"
echo "Using $FP32_MODEL as base"
echo "Dataset: $DATASET at $DATASET_ROOT"
echo "8-bit pre-finetune epochs: $UNIFORM_8BIT_EPOCHS"
echo "Force first/last layer high precision: $FORCE_FIRST_LAST_LAYER"
echo "Learning rate: $LEARNING_RATE"
echo "GPU ID: $GPU_ID"
echo "W&B logging: $WANDB_ENABLE"
if [ "$WANDB_ENABLE" = "true" ]; then
  echo "W&B project: $WANDB_PROJECT"
fi
echo "Repository root: $REPO_ROOT"
echo "========================================================="

# Check if FP32 model file exists
if [ -n "$DATASET_SUFFIX" ]; then
  FP32_MODEL_FILE="${REPO_ROOT}/checkpoints/${FP32_MODEL}_pretrained_${DATASET_SUFFIX}/model_best.pth.tar"
else
  FP32_MODEL_FILE="${REPO_ROOT}/checkpoints/${FP32_MODEL}_pretrained/model_best.pth.tar"
fi

if [ ! -f "$FP32_MODEL_FILE" ]; then
  echo "Error: FP32 fine-tuned model not found at $FP32_MODEL_FILE"
  echo "Please run FP32 pretraining first."
  exit 1
fi

# Step 1: Generate uniform 8-bit quantization strategy
echo ""
echo "Step 1/3: Generating uniform 8-bit quantization strategy..."
mkdir -p "${REPO_ROOT}/save/uniform_strategies"
python "${REPO_ROOT}/utils/create_uniform_strategy.py" --arch $QUANT_MODEL --w_bit 8 --a_bit 8 $FORCE_FIRST_LAST_CLI_ARG --output "${REPO_ROOT}/save/uniform_strategies"

# Check if the command succeeded
if [ $? -ne 0 ]; then
  echo "Error: Failed to create uniform 8-bit strategy file."
  exit 1
fi

UNIFORM_STRATEGY_FILE="${REPO_ROOT}/save/uniform_strategies/${QUANT_MODEL}_w8a8.npy"
if [ ! -f "$UNIFORM_STRATEGY_FILE" ]; then
  echo "Error: Failed to generate uniform 8-bit strategy file."
  exit 1
fi

# Step 2: Fine-tune the model with uniform 8-bit quantization
echo ""
echo "Step 2/3: Fine-tuning the model with uniform 8-bit quantization..."
if [ -n "$DATASET_SUFFIX" ]; then
  UNIFORM_MODEL_DIR="${REPO_ROOT}/checkpoints/${QUANT_MODEL}_per-tensor_uniform_8bit_${DATASET_SUFFIX}"
else
  UNIFORM_MODEL_DIR="${REPO_ROOT}/checkpoints/${QUANT_MODEL}_per-tensor_uniform_8bit"
fi
mkdir -p $UNIFORM_MODEL_DIR

# Create symlink to FP32 model for pretrained initialization
echo "Creating symlink to FP32 model for 8-bit quantization..."
ln -sf "$FP32_MODEL_FILE" "${REPO_ROOT}/pretrained/imagenet/${QUANT_MODEL}.pth.tar"

# Run fine-tuning with uniform 8-bit quantization
python "${REPO_ROOT}/finetune.py" \
  -a $QUANT_MODEL \
  -d $DATASET_ROOT \
  --data_name $DATASET \
  --epochs $UNIFORM_8BIT_EPOCHS \
  --lr $LEARNING_RATE \
  --lr_type cos \
  --wd 0.0001 \
  --batch_size $BATCH_SIZE \
  --workers $NUM_WORKERS \
  --pretrained \
  --checkpoint $UNIFORM_MODEL_DIR \
  --strategy_file $UNIFORM_STRATEGY_FILE \
  --amp \
  --gpu_id $GPU_ID \
  $WANDB_CLI_ARG \
  --wandb_project "$WANDB_PROJECT"

# Check if training succeeded
if [ $? -ne 0 ]; then
  echo "Error: 8-bit quantization fine-tuning failed."
  exit 1
fi

# Check if 8-bit model exists
UNIFORM_MODEL_FILE="${UNIFORM_MODEL_DIR}/model_best.pth.tar"
if [ ! -f "$UNIFORM_MODEL_FILE" ]; then
  echo "Error: 8-bit fine-tuned model not found. Fine-tuning may have failed."
  exit 1
fi

# Step 3: Evaluate the 8-bit quantized model
echo ""
echo "Step 3/3: Evaluating the 8-bit quantized model..."
# Run the evaluation, display output to console and capture it
UNIFORM_EVAL_OUTPUT=$(python "${REPO_ROOT}/finetune.py" \
    -a $QUANT_MODEL \
    -d $DATASET_ROOT \
    --data_name $DATASET \
    --evaluate \
    --batch_size $BATCH_SIZE \
    --workers $NUM_WORKERS \
    --resume $UNIFORM_MODEL_FILE \
    --amp \
    --gpu_id $GPU_ID \
  --strategy_file $UNIFORM_STRATEGY_FILE 2>&1 | tee /dev/tty)

# Check if evaluation succeeded
if [ $? -ne 0 ]; then
  echo "Error: 8-bit model evaluation failed."
  exit 1
fi

# Extract the 8-bit model accuracy
UNIFORM_8BIT_ACCURACY=$(echo "$UNIFORM_EVAL_OUTPUT" | grep -oP "Test Acc:\s+\K[0-9\.]+")
UNIFORM_8BIT_ACCURACY5=$(echo "$UNIFORM_EVAL_OUTPUT" | grep -oP "Test Acc5:\s+\K[0-9\.]+")
echo "Uniform 8-bit model accuracy: $UNIFORM_8BIT_ACCURACY% (Top-5: $UNIFORM_8BIT_ACCURACY5%)"

echo ""
echo "========================================================="
echo "INT8 pretraining complete!"
echo "Uniform 8-bit model: $UNIFORM_MODEL_FILE"
echo "Uniform 8-bit accuracy: $UNIFORM_8BIT_ACCURACY% (Top-5: $UNIFORM_8BIT_ACCURACY5%)"
echo "========================================================="
