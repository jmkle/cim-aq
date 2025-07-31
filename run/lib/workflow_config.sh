#!/bin/bash

# Configuration management functions for CIM-AQ workflows
# This library provides functions for parsing YAML configs and setting up workflow variables

# Load configuration from YAML file
load_workflow_config() {
  local config_file="$1"
  local repo_root="$2"
  local script_dir="$3"

  # Resolve config file path
  if [[ "$config_file" != /* ]]; then
    config_file="${repo_root}/${config_file}"
  fi

  # Check if config file exists
  if [ ! -f "$config_file" ]; then
    echo "Error: Configuration file '$config_file' not found" >&2
    return 1
  fi

  # Parse YAML configuration using Python helper
  local config_parser="${script_dir}/parse_config.py"
  if [ ! -f "$config_parser" ]; then
    echo "Error: Configuration parser '$config_parser' not found" >&2
    return 1
  fi

  # Load configuration variables into current shell
  eval $(python3 "$config_parser" "$config_file" --repo-root "$repo_root")
  return $?
}

# Setup workflow paths and variables
setup_workflow_variables() {
  local repo_root="$1"

  # Check if both datasets are the same - if so, skip large dataset stage
  if [ "$SMALL_DATASET" = "$LARGE_DATASET" ] && [ "$SMALL_DATASET_ROOT" = "$LARGE_DATASET_ROOT" ] && [ "$ENABLE_LARGE_DATASET" = "true" ]; then
    echo "⚠️  Small and large datasets are identical - disabling large dataset stage"
    ENABLE_LARGE_DATASET="false"
  fi

  OUTPUT_SUFFIX="accdrop${MAX_ACCURACY_DROP}bit${MIN_BIT}${MAX_BIT}"

  # Construct file paths for small dataset
  SMALL_INT8_MODEL_FILE="${repo_root}/checkpoints/${QUANT_MODEL}_per-tensor_uniform_8bit_${SMALL_DATASET}/model_best.pth.tar"
  SMALL_FP32_MODEL_FILE="${repo_root}/checkpoints/${FP32_MODEL}_pretrained_${SMALL_DATASET}/model_best.pth.tar"

  # Construct file paths for large dataset (if enabled)
  if [ "$ENABLE_LARGE_DATASET" = "true" ]; then
    LARGE_INT8_MODEL_FILE="${repo_root}/checkpoints/${QUANT_MODEL}_per-tensor_uniform_8bit_${LARGE_DATASET}/model_best.pth.tar"
    LARGE_FP32_MODEL_FILE="${repo_root}/checkpoints/${FP32_MODEL}_pretrained_${LARGE_DATASET}/model_best.pth.tar"
  fi

  # Construct prefixed output names for RL and mixed precision phases
  if [ -n "$OUTPUT_PREFIX" ]; then
    SMALL_RL_OUTPUT_SUFFIX="${OUTPUT_PREFIX}_${OUTPUT_SUFFIX}_${SMALL_DATASET}"
    SMALL_MP_OUTPUT_SUFFIX="${OUTPUT_PREFIX}_mixed_precision_${SMALL_DATASET}"
    if [ "$ENABLE_LARGE_DATASET" = "true" ]; then
      LARGE_MP_OUTPUT_SUFFIX="${OUTPUT_PREFIX}_mixed_precision_${LARGE_DATASET}"
    fi
  else
    SMALL_RL_OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_${SMALL_DATASET}"
    SMALL_MP_OUTPUT_SUFFIX="mixed_precision_${SMALL_DATASET}"
    if [ "$ENABLE_LARGE_DATASET" = "true" ]; then
      LARGE_MP_OUTPUT_SUFFIX="mixed_precision_${LARGE_DATASET}"
    fi
  fi
}

# Print configuration summary
print_workflow_config() {
  local config_file="$1"
  local repo_root="$2"

  echo ""
  echo "========================================================="
  echo "Configuration Summary:"
  echo "========================================================="
  echo "📋 EXPERIMENT CONFIGURATION:"
  echo "├─ Configuration file: $config_file"
  echo "├─ Quantized model: $QUANT_MODEL"
  echo "├─ FP32 baseline model: $FP32_MODEL"
  echo "├─ Output prefix: ${OUTPUT_PREFIX:-None}"
  echo "└─ Repository root: $repo_root"
  echo ""
  echo "📊 DATASET CONFIGURATION:"
  echo "├─ Stage 1 (Small): $SMALL_DATASET at $SMALL_DATASET_ROOT"
  if [ "$ENABLE_LARGE_DATASET" = "true" ]; then
    echo "├─ Stage 2 (Large): $LARGE_DATASET at $LARGE_DATASET_ROOT"
    echo "└─ Large dataset stage: ENABLED"
  else
    echo "└─ Large dataset stage: DISABLED"
  fi
  echo ""
  echo "⚙️  QUANTIZATION CONFIGURATION:"
  echo "├─ Maximum accuracy drop: $MAX_ACCURACY_DROP%"
  echo "├─ Bit-width range: $MIN_BIT-$MAX_BIT bits"
  echo "├─ Force first/last layer high precision: $FORCE_FIRST_LAST_LAYER"
  echo "└─ Consider cell resolution: $CONSIDER_CELL_RESOLUTION"
  echo ""
  echo "🏋️  TRAINING CONFIGURATION:"
  echo "├─ RL training episodes: $TRAIN_EPISODES"
  echo "├─ Small dataset learning rates:"
  echo "│  ├─ FP32 pretraining: $SMALL_FP32_LEARNING_RATE"
  echo "│  ├─ INT8 pretraining: $SMALL_INT8_LEARNING_RATE"
  echo "│  ├─ Mixed precision finetuning: $SMALL_MP_LEARNING_RATE"
  echo "│  └─ RL search finetune: $SMALL_RL_FINETUNE_LEARNING_RATE"
  if [ "$ENABLE_LARGE_DATASET" = "true" ]; then
    echo "├─ Large dataset learning rates:"
    echo "│  ├─ FP32 pretraining: $LARGE_FP32_LEARNING_RATE"
    echo "│  ├─ INT8 pretraining: $LARGE_INT8_LEARNING_RATE"
    echo "│  └─ Mixed precision finetuning: $LARGE_MP_LEARNING_RATE"
  fi
  echo "├─ Small dataset epochs:"
  echo "│  ├─ FP32: $SMALL_FP32_EPOCHS"
  echo "│  ├─ INT8: $SMALL_8BIT_EPOCHS"
  echo "│  ├─ Search finetune: $SMALL_SEARCH_FINETUNE_EPOCHS"
  echo "│  └─ Final finetune: $SMALL_FINETUNE_EPOCHS"
  if [ "$ENABLE_LARGE_DATASET" = "true" ]; then
    echo "└─ Large dataset epochs:"
    echo "   ├─ FP32: $LARGE_FP32_EPOCHS"
    echo "   ├─ INT8: $LARGE_8BIT_EPOCHS"
    echo "   └─ Final finetune: $LARGE_FINETUNE_EPOCHS"
  else
    echo "└─ Large dataset training: DISABLED"
  fi
  echo ""
  echo "📁 MODEL FILE PATHS:"
  echo "├─ Small dataset INT8: $SMALL_INT8_MODEL_FILE"
  echo "├─ Small dataset FP32: $SMALL_FP32_MODEL_FILE"
  if [ "$ENABLE_LARGE_DATASET" = "true" ]; then
    echo "├─ Large dataset INT8: $LARGE_INT8_MODEL_FILE"
    echo "└─ Large dataset FP32: $LARGE_FP32_MODEL_FILE"
  else
    echo "└─ Large dataset models: N/A (stage disabled)"
  fi
  echo ""
  echo "📊 LOGGING CONFIGURATION:"
  if [ "$WANDB_ENABLE" = "true" ]; then
    echo "├─ W&B logging: ENABLED"
    echo "└─ W&B project: $WANDB_PROJECT"
  else
    echo "└─ W&B logging: DISABLED"
  fi
  echo "========================================================="
}
