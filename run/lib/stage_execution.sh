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

# Stage execution functions for CIM-AQ workflows
# This library provides functions for executing individual workflow stages

# Source cleanup utilities
if ! source "${SCRIPT_DIR}/lib/cleanup_utils.sh"; then
  echo "❌ Failed to load cleanup_utils.sh"
  exit 1
fi

# Execute a single stage with the given parameters
execute_stage() {
  local stage_name="$1"      # "Stage 1" or "Stage 2"
  local dataset="$2"
  local dataset_root="$3"
  local script_dir="$4"
  local repo_root="$5"
  local skip_rl="${6:-false}"  # Skip RL search for Stage 2
  local batch_size="${7:-256}"
  local num_workers="${8:-32}"

  echo ""
  if [ "$skip_rl" = "true" ]; then
    echo "🔄 $stage_name: Dataset ($dataset) - Apply Discovered Policy"
  else
    echo "🔄 $stage_name: Dataset ($dataset)"
  fi
  echo "========================================================="

  # Determine variable prefixes based on stage
  local prefix
  if [ "$stage_name" = "Stage 1" ]; then
    prefix="SMALL"
  else
    prefix="LARGE"
  fi

  local fp32_model_var="${prefix}_FP32_MODEL_FILE"
  local int8_model_var="${prefix}_INT8_MODEL_FILE"
  local fp32_epochs_var="${prefix}_FP32_EPOCHS"
  local int8_epochs_var="${prefix}_8BIT_EPOCHS"
  local finetune_epochs_var="${prefix}_FINETUNE_EPOCHS"
  local rl_output_var="${prefix}_RL_OUTPUT_SUFFIX"
  local mp_output_var="${prefix}_MP_OUTPUT_SUFFIX"

  local fp32_model_file="${!fp32_model_var}"
  local int8_model_file="${!int8_model_var}"
  local fp32_epochs="${!fp32_epochs_var}"
  local int8_epochs="${!int8_epochs_var}"
  local finetune_epochs="${!finetune_epochs_var}"
  local rl_output_suffix="${!rl_output_var}"
  local mp_output_suffix="${!mp_output_var}"

  # Determine dataset-specific learning rates
  local fp32_lr_var="${prefix}_FP32_LEARNING_RATE"
  local int8_lr_var="${prefix}_INT8_LEARNING_RATE"
  local mp_lr_var="${prefix}_MP_LEARNING_RATE"

  local fp32_lr="${!fp32_lr_var}"
  local int8_lr="${!int8_lr_var}"
  local mp_lr="${!mp_lr_var}"

  # For RL search, always use small dataset learning rate since RL only runs on small dataset
  local rl_finetune_lr="$SMALL_RL_FINETUNE_LEARNING_RATE"

  # Check what models already exist
  local skip_fp32=false
  local skip_int8=false

  if [ -f "$int8_model_file" ]; then
    echo "✅ Dataset INT8 model found at $int8_model_file - skipping FP32 and INT8 pretraining"
    skip_fp32=true
    skip_int8=true
  elif [ -f "$fp32_model_file" ]; then
    echo "✅ Dataset FP32 model found at $fp32_model_file - skipping FP32 pretraining"
    skip_fp32=true
  else
    echo "🔄 No pretrained models found for dataset - will run full workflow"
  fi

  # Step X.1: FP32 pretraining (if needed)
  if [ "$skip_fp32" = false ]; then
    echo ""
    echo "========== $stage_name.1: FP32 pretraining on $dataset =========="
    bash "${script_dir}/run_fp32_pretraining.sh" \
      "$FP32_MODEL" \
      "$dataset" \
      "$dataset_root" \
      "$fp32_epochs" \
      "${dataset}" \
      "$fp32_lr" \
      "$WANDB_ENABLE" \
      "$WANDB_PROJECT" \
      "$GPU_ID" \
      "$batch_size" \
      "$num_workers"

    if [ $? -ne 0 ]; then
      echo "Error: $stage_name FP32 pretraining failed"
      return 1
    fi

    # Register the FP32 checkpoint directory for safe cleanup
    fp32_checkpoint_dir="${repo_root}/checkpoints/${FP32_MODEL}_pretrained_${dataset}"
    register_checkpoint_dir "$fp32_checkpoint_dir"

    # Step-level cleanup after FP32 training
    if [ "${CLEANUP_FREQUENCY:-end}" = "step" ]; then
      cleanup_intermediate_files "FP32 pretraining" "$repo_root"
    fi
  else
    echo "⏭️  Skipping $stage_name FP32 pretraining (model exists)"
  fi

  # Step X.2: INT8 pretraining (if needed)
  if [ "$skip_int8" = false ]; then
    echo ""
    echo "========== $stage_name.2: INT8 pretraining on $dataset =========="
    bash "${script_dir}/run_int8_pretraining.sh" \
      "$QUANT_MODEL" \
      "$FP32_MODEL" \
      "$dataset" \
      "$dataset_root" \
      "$int8_epochs" \
      "$FORCE_FIRST_LAST_LAYER" \
      "${dataset}" \
      "$int8_lr" \
      "$WANDB_ENABLE" \
      "$WANDB_PROJECT" \
      "$GPU_ID" \
      "$batch_size" \
      "$num_workers"

    if [ $? -ne 0 ]; then
      echo "Error: $stage_name INT8 pretraining failed"
      return 1
    fi

    # Register the INT8 checkpoint directory for safe cleanup
    int8_checkpoint_dir="${repo_root}/checkpoints/${QUANT_MODEL}_per-tensor_uniform_8bit_${dataset}"
    register_checkpoint_dir "$int8_checkpoint_dir"

    # Step-level cleanup after INT8 training
    if [ "${CLEANUP_FREQUENCY:-end}" = "step" ]; then
      cleanup_intermediate_files "INT8 pretraining" "$repo_root"
    fi
  else
    echo "⏭️  Skipping $stage_name INT8 pretraining (model exists)"
  fi

  # Step X.3: RL-based quantization search (Stage 1 only)
  local strategy_file
  if [ "$skip_rl" = "false" ]; then
    echo ""
    echo "========== $stage_name.3: RL-based quantization search on $dataset =========="
    bash "${script_dir}/run_rl_quantize.sh" \
      "$QUANT_MODEL" \
      "$dataset" \
      "$dataset_root" \
      "$MAX_ACCURACY_DROP" \
      "$MIN_BIT" \
      "$MAX_BIT" \
      "$TRAIN_EPISODES" \
      "$SMALL_SEARCH_FINETUNE_EPOCHS" \
      "$FORCE_FIRST_LAST_LAYER" \
      "$CONSIDER_CELL_RESOLUTION" \
      "$rl_output_suffix" \
      "$rl_finetune_lr" \
      "$int8_model_file" \
      "$WANDB_ENABLE" \
      "$WANDB_PROJECT" \
      "$GPU_ID" \
      "$batch_size" \
      "$num_workers"

    if [ $? -ne 0 ]; then
      echo "Error: $stage_name RL quantization search failed"
      return 1
    fi

    # Step-level cleanup after RL search
    if [ "${CLEANUP_FREQUENCY:-end}" = "step" ]; then
      cleanup_intermediate_files "RL quantization search" "$repo_root"
    fi

    strategy_file="${repo_root}/save/${QUANT_MODEL}_${dataset}_${rl_output_suffix}_from_8bit/best_policy.npy"
  else
    # Use discovered strategy from Stage 1
    strategy_file="$DISCOVERED_STRATEGY_FILE"
    echo ""
    echo "========== $stage_name.3: Using discovered strategy =========="
    echo "Strategy file: $strategy_file"

    if [ ! -f "$strategy_file" ]; then
      echo "❌ Error: No strategy file found from Stage 1. Cannot proceed with $stage_name."
      return 1
    fi
  fi

  # Step X.4: Mixed precision fine-tuning
  echo ""
  echo "========== $stage_name.4: Mixed precision fine-tuning on $dataset =========="
  bash "${script_dir}/run_mp_finetuning.sh" \
    "$QUANT_MODEL" \
    "$dataset" \
    "$dataset_root" \
    "$finetune_epochs" \
    "$strategy_file" \
    "$mp_output_suffix" \
    "$mp_lr" \
    "$int8_model_file" \
    "$WANDB_ENABLE" \
    "$WANDB_PROJECT" \
    "$GPU_ID" \
    "$batch_size" \
    "$num_workers"

  if [ $? -ne 0 ]; then
    echo "Error: $stage_name mixed precision fine-tuning failed"
    return 1
  fi

  # Register the mixed precision checkpoint directory for safe cleanup
  mp_checkpoint_dir="${repo_root}/checkpoints/${QUANT_MODEL}_${mp_output_suffix}"
  register_checkpoint_dir "$mp_checkpoint_dir"

  # Step-level cleanup after mixed precision fine-tuning
  if [ "${CLEANUP_FREQUENCY:-end}" = "step" ]; then
    cleanup_intermediate_files "Mixed precision fine-tuning" "$repo_root"
  fi

  # Set strategy and final model file for evaluation
  if [ "$stage_name" = "Stage 1" ]; then
    SMALL_STRATEGY_FILE="$strategy_file"
    SMALL_FINAL_MODEL_FILE="${repo_root}/checkpoints/${QUANT_MODEL}_${mp_output_suffix}/model_best.pth.tar"
    DISCOVERED_STRATEGY_FILE="$strategy_file"  # Save for Stage 2
  else
    LARGE_STRATEGY_FILE="$strategy_file"
    LARGE_FINAL_MODEL_FILE="${repo_root}/checkpoints/${QUANT_MODEL}_${mp_output_suffix}/model_best.pth.tar"
  fi

  return 0
}
