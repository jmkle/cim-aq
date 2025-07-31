#!/bin/bash

# Get the directory of the script and the repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# This script handles mixed precision fine-tuning:
# 1. Fine-tune the model with the best mixed precision strategy
# 2. Evaluate the final quantized model

# Usage: bash run_mp_finetuning.sh [quant_model] [dataset] [dataset_root] [finetune_epochs] [strategy_file] [output_suffix] [learning_rate] [uniform_model_file] [wandb_enable] [wandb_project] [gpu_id]
# Example: bash run_mp_finetuning.sh qvgg16 imagenet100 /path/to/dataset 30 /path/to/strategy.npy custom_run 0.0005 /path/to/uniform/model.pth.tar false cim-aq-quantization 1

# Default values
QUANT_MODEL=${1:-"qvgg16"}                          # Quantized model architecture
DATASET=${2:-"imagenet100"}                         # Dataset name
DATASET_ROOT=${3:-"${REPO_ROOT}/data/imagenet100"}  # Path to dataset
FINETUNE_EPOCHS=${4:-"30"}                          # Number of epochs for final fine-tuning
STRATEGY_FILE=${5:-""}                              # Path to strategy file
OUTPUT_SUFFIX=${6:-"mixed_precision"}               # Output suffix for checkpoint directory
LEARNING_RATE=${7:-"0.0005"}                        # Learning rate for mixed precision fine-tuning
UNIFORM_MODEL_FILE=${8:-"${REPO_ROOT}/checkpoints/${QUANT_MODEL}_per-tensor_uniform_8bit/model_best.pth.tar"} # Path to INT8 model for initialization
WANDB_ENABLE=${9:-"false"}                          # Enable W&B logging
WANDB_PROJECT=${10:-"cim-aq-quantization"}          # W&B project name
GPU_ID=${11:-"1"}                                   # GPU ID(s) for CUDA_VISIBLE_DEVICES

if [[ "$WANDB_ENABLE" == "true" || "$WANDB_ENABLE" == "True" || "$WANDB_ENABLE" == "1" ]]; then
  WANDB_CLI_ARG="--wandb_enable"
else
  WANDB_CLI_ARG=""
fi

echo "========================================================="
echo "Starting mixed precision fine-tuning for $QUANT_MODEL"
echo "Dataset: $DATASET at $DATASET_ROOT"
echo "Fine-tuning epochs: $FINETUNE_EPOCHS"
echo "Strategy file: $STRATEGY_FILE"
echo "Output suffix: $OUTPUT_SUFFIX"
echo "Learning rate: $LEARNING_RATE"
echo "GPU ID: $GPU_ID"
echo "W&B logging: $WANDB_ENABLE"
if [ "$WANDB_ENABLE" = "true" ]; then
  echo "W&B project: $WANDB_PROJECT"
fi
echo "Repository root: $REPO_ROOT"
echo "========================================================="

# Auto-detect strategy file if not provided
if [ -z "$STRATEGY_FILE" ] || [ ! -f "$STRATEGY_FILE" ]; then
  echo "Strategy file not provided or doesn't exist. Trying to auto-detect..."
  # Look for the most recent strategy file
  STRATEGY_PATTERN="${REPO_ROOT}/save/${QUANT_MODEL}_${DATASET}_*_from_8bit/best_policy.npy"
  STRATEGY_FILE=$(ls -t $STRATEGY_PATTERN 2>/dev/null | head -n1)

  if [ -z "$STRATEGY_FILE" ] || [ ! -f "$STRATEGY_FILE" ]; then
    echo "Error: No strategy file found. Please run RL quantization first."
    exit 1
  fi
  echo "Auto-detected strategy file: $STRATEGY_FILE"
fi

# Check if INT8 model exists (used as base for mixed precision)
if [ ! -f "$UNIFORM_MODEL_FILE" ]; then
  echo "Error: INT8 fine-tuned model not found at $UNIFORM_MODEL_FILE"
  echo "Please run INT8 pretraining first."
  exit 1
fi

# Step 1: Fine-tune the model with the best strategy
echo ""
echo "Step 1/2: Fine-tuning the model with the best bit-width strategy..."

# Create symlink to 8-bit model for pretrained initialization of mixed precision model
echo "Creating symlink to 8-bit model for mixed precision fine-tuning..."
mkdir -p "${REPO_ROOT}/pretrained/imagenet"
ln -sf "$UNIFORM_MODEL_FILE" "${REPO_ROOT}/pretrained/imagenet/${QUANT_MODEL}.pth.tar"

CHECKPOINT_DIR="${REPO_ROOT}/checkpoints/${QUANT_MODEL}_${OUTPUT_SUFFIX}"
mkdir -p "$CHECKPOINT_DIR"

python "${REPO_ROOT}/finetune.py" \
  -a $QUANT_MODEL \
  -d $DATASET_ROOT \
  --data_name $DATASET \
  --epochs $FINETUNE_EPOCHS \
  --lr $LEARNING_RATE \
  --lr_type cos \
  --wd 0.0001 \
  --train_batch 256 \
  --test_batch 512 \
  --workers 32 \
  --pretrained \
  --checkpoint "$CHECKPOINT_DIR" \
  --amp \
  --gpu_id $GPU_ID \
  --strategy_file $STRATEGY_FILE \
  $WANDB_CLI_ARG \
  --wandb_project "$WANDB_PROJECT"

# Check if fine-tuned model exists
FINAL_MODEL_FILE="${CHECKPOINT_DIR}/model_best.pth.tar"
if [ ! -f "$FINAL_MODEL_FILE" ]; then
  echo "Error: Final fine-tuned model not found. Fine-tuning may have failed."
  exit 1
fi

# Step 2: Evaluate the quantized model
echo ""
echo "Step 2/2: Evaluating the final quantized model..."
# Run the evaluation, display output to console and capture it
FINAL_EVAL_OUTPUT=$(python "${REPO_ROOT}/finetune.py" \
    -a $QUANT_MODEL \
    -d $DATASET_ROOT \
    --data_name $DATASET \
    --evaluate \
    --resume $FINAL_MODEL_FILE \
    --gpu_id $GPU_ID \
    --amp \
  --strategy_file $STRATEGY_FILE 2>&1 | tee /dev/tty)

# Try to extract the final accuracy
FINAL_ACCURACY=$(echo "$FINAL_EVAL_OUTPUT" | grep -oP "Test Acc:\s+\K[0-9\.]+")
FINAL_ACCURACY5=$(echo "$FINAL_EVAL_OUTPUT" | grep -oP "Test Acc5:\s+\K[0-9\.]+")

echo ""
echo "========================================================="
echo "Mixed precision fine-tuning complete!"
echo "Final mixed precision model: $FINAL_MODEL_FILE"
echo "Final accuracy with mixed precision: $FINAL_ACCURACY% (Top-5: $FINAL_ACCURACY5%)"
echo "========================================================="
