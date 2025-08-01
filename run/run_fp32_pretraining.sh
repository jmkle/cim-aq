#!/bin/bash
##############################################################################
# Copyright (C) 2025 Joel Klein                                              #
# All Rights Reserved                                                        #
#                                                                            #
# This work is licensed under the terms described in the LICENSE file        #
# found in the root directory of this source tree.                           #
##############################################################################

# Get the directory of the script and the repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# This script handles FP32 model pretraining:
# 1. Download pretrained weights from torchvision
# 2. Create a dummy strategy file for FP32 model evaluation
# 3. Fine-tune the FP32 model to adjust the last layer for new classes
# 4. Evaluate the pretrained model to get baseline accuracy

# Usage: bash run_fp32_pretraining.sh [fp32_model] [dataset] [dataset_root] [fp32_finetune_epochs] [dataset_suffix] [learning_rate] [wandb_enable] [wandb_project] [gpu_id]
# Example: bash run_fp32_pretraining.sh custom_vgg16 imagenet100 /path/to/dataset 5 imagenet100 0.0005 false cim-aq-quantization 1

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

BASE_MODEL_NAME=${FP32_MODEL/custom_/}

# Determine the number of classes based on the dataset
if [ "$DATASET" = "imagenet100" ]; then
  NUM_CLASSES=100
elif [ "$DATASET" = "imagenet" ]; then
  NUM_CLASSES=1000
else
  echo "Unknown dataset: $DATASET - defaulting to 1000 classes"
  NUM_CLASSES=1000
fi

# Convert W&B enable string to CLI argument
if [[ "$WANDB_ENABLE" == "true" || "$WANDB_ENABLE" == "True" || "$WANDB_ENABLE" == "1" ]]; then
  WANDB_CLI_ARG="--wandb_enable"
else
  WANDB_CLI_ARG=""
fi

echo "========================================================="
echo "Starting FP32 model pretraining for $FP32_MODEL"
echo "Dataset: $DATASET at $DATASET_ROOT"
echo "FP32 fine-tune epochs: $FP32_FINETUNE_EPOCHS"
echo "Number of classes: $NUM_CLASSES"
echo "Learning rate: $LEARNING_RATE"
echo "GPU ID: $GPU_ID"
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
  --train_batch 256 \
  --test_batch 512 \
  --workers 32 \
  --pretrained \
  --checkpoint $FP32_MODEL_DIR \
  --strategy_file $FP32_STRATEGY_FILE \
  --gpu_id $GPU_ID \
  $WANDB_CLI_ARG \
  --wandb_project "$WANDB_PROJECT"

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
EVAL_OUTPUT=$(python "${REPO_ROOT}/finetune.py" \
    -a $FP32_MODEL \
    -d $DATASET_ROOT \
    --data_name $DATASET \
    --evaluate \
    --test_batch 256 \
    --workers 32 \
    --strategy_file $FP32_STRATEGY_FILE \
    --gpu_id $GPU_ID \
  --resume $FP32_MODEL_FILE 2>&1 | tee /dev/tty)

# Store the baseline accuracy by parsing the output
BASELINE_ACCURACY=$(echo "$EVAL_OUTPUT" | grep -oP "Test Acc:\s+\K[0-9\.]+")
BASELINE_ACCURACY5=$(echo "$EVAL_OUTPUT" | grep -oP "Test Acc5:\s+\K[0-9\.]+")
echo "Baseline accuracy with fine-tuned FP32 model: $BASELINE_ACCURACY% (Top-5: $BASELINE_ACCURACY5%)"

echo ""
echo "========================================================="
echo "FP32 pretraining complete!"
echo "FP32 fine-tuned model: $FP32_MODEL_FILE"
echo "Baseline accuracy: $BASELINE_ACCURACY% (Top-5: $BASELINE_ACCURACY5%)"
echo "========================================================="
