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

# Model evaluation and analysis functions for CIM-AQ workflows
# This library provides functions for evaluating models and extracting accuracy metrics

# Evaluate a model and extract accuracy metrics
evaluate_model() {
  local model_arch="$1"
  local model_file="$2"
  local dataset_root="$3"
  local dataset_name="$4"
  local strategy_file="$5"
  local repo_root="$6"
  local use_amp="${7:-false}"
  local batch_size="${8:-256}"
  local num_workers="${9:-32}"

  if [ ! -f "$model_file" ]; then
    echo "Warning: Model file not found: $model_file" >&2
    return 1
  fi

  local eval_cmd="python ${repo_root}/finetune.py \
        -a $model_arch \
        -d $dataset_root \
        --data_name $dataset_name \
        --evaluate \
        --batch_size $batch_size \
        --workers $num_workers \
        --strategy_file $strategy_file \
        --resume $model_file"

  if [ "$use_amp" = "true" ]; then
    eval_cmd="$eval_cmd --amp"
  fi

  local eval_output
  eval_output=$($eval_cmd 2>&1)

  if [ $? -ne 0 ]; then
    echo "Error: Model evaluation failed for $model_file" >&2
    return 1
  fi

  # Extract accuracy metrics
  local accuracy=$(echo "$eval_output" | grep -oP "Test Acc:\s+\K[0-9\.]+")
  local accuracy5=$(echo "$eval_output" | grep -oP "Test Acc5:\s+\K[0-9\.]+")

  # Export results to calling script
  echo "ACCURACY=${accuracy:-Unknown}"
  echo "ACCURACY5=${accuracy5:-Unknown}"
  echo "EVAL_OUTPUT=$eval_output"
}

# Evaluate all models for a given dataset stage
evaluate_stage_models() {
  local stage_name="$1"  # "SMALL" or "LARGE"
  local dataset="$2"
  local dataset_root="$3"
  local repo_root="$4"
  local batch_size="${5:-256}"
  local num_workers="${6:-32}"

  local fp32_model_var="${stage_name}_FP32_MODEL_FILE"
  local int8_model_var="${stage_name}_INT8_MODEL_FILE"
  local final_model_var="${stage_name}_FINAL_MODEL_FILE"
  local strategy_var="${stage_name}_STRATEGY_FILE"

  local fp32_model_file="${!fp32_model_var}"
  local int8_model_file="${!int8_model_var}"
  local final_model_file="${!final_model_var}"
  local strategy_file="${!strategy_var}"

  local evaluation_failed=false

  # Evaluate FP32 baseline model
  if [ -f "$fp32_model_file" ]; then
    local fp32_strategy_file="${repo_root}/save/uniform_strategies/${FP32_MODEL}_w8a8.npy"
    local fp32_results
    fp32_results=$(evaluate_model "$FP32_MODEL" "$fp32_model_file" "$dataset_root" "$dataset" "$fp32_strategy_file" "$repo_root" "false" "$batch_size" "$num_workers")

    if [ $? -eq 0 ]; then
      eval "$fp32_results"
      declare -g "${stage_name}_BASELINE_ACCURACY=$ACCURACY"
      declare -g "${stage_name}_BASELINE_ACCURACY5=$ACCURACY5"
    else
      echo "Warning: FP32 model evaluation failed for $stage_name" >&2
      evaluation_failed=true
    fi
  fi

  # Evaluate INT8 model
  if [ -f "$int8_model_file" ]; then
    local uniform_strategy_file="${repo_root}/save/uniform_strategies/${QUANT_MODEL}_w8a8.npy"
    local int8_results
    int8_results=$(evaluate_model "$QUANT_MODEL" "$int8_model_file" "$dataset_root" "$dataset" "$uniform_strategy_file" "$repo_root" "true" "$batch_size" "$num_workers")

    if [ $? -eq 0 ]; then
      eval "$int8_results"
      declare -g "${stage_name}_UNIFORM_8BIT_ACCURACY=$ACCURACY"
      declare -g "${stage_name}_UNIFORM_8BIT_ACCURACY5=$ACCURACY5"
    else
      echo "Warning: INT8 model evaluation failed for $stage_name" >&2
      evaluation_failed=true
    fi
  fi

  # Evaluate mixed precision model
  if [ -f "$final_model_file" ] && [ -f "$strategy_file" ]; then
    local final_results
    final_results=$(evaluate_model "$QUANT_MODEL" "$final_model_file" "$dataset_root" "$dataset" "$strategy_file" "$repo_root" "true" "$batch_size" "$num_workers")

    if [ $? -eq 0 ]; then
      eval "$final_results"
      declare -g "${stage_name}_FINAL_ACCURACY=$ACCURACY"
      declare -g "${stage_name}_FINAL_ACCURACY5=$ACCURACY5"
    else
      echo "Warning: Mixed precision model evaluation failed for $stage_name" >&2
      evaluation_failed=true
    fi
  fi

  # Return error if any evaluation failed
  if [ "$evaluation_failed" = true ]; then
    return 1
  fi

  return 0
}

# Print stage results summary
print_stage_results() {
  local stage_name="$1"  # "Stage 1" or "Stage 2"
  local dataset="$2"

  local prefix
  if [ "$stage_name" = "Stage 1" ]; then
    prefix="SMALL"
  else
    prefix="LARGE"
  fi

  local fp32_model_var="${prefix}_FP32_MODEL_FILE"
  local int8_model_var="${prefix}_INT8_MODEL_FILE"
  local final_model_var="${prefix}_FINAL_MODEL_FILE"
  local strategy_var="${prefix}_STRATEGY_FILE"
  local baseline_acc_var="${prefix}_BASELINE_ACCURACY"
  local baseline_acc5_var="${prefix}_BASELINE_ACCURACY5"
  local uniform_acc_var="${prefix}_UNIFORM_8BIT_ACCURACY"
  local uniform_acc5_var="${prefix}_UNIFORM_8BIT_ACCURACY5"
  local final_acc_var="${prefix}_FINAL_ACCURACY"
  local final_acc5_var="${prefix}_FINAL_ACCURACY5"

  echo ""
  echo "========== $stage_name Results ($dataset) =========="
  echo "FP32 fine-tuned model: ${!fp32_model_var}"
  echo "Baseline accuracy: ${!baseline_acc_var:-Unknown}% (Top-5: ${!baseline_acc5_var:-Unknown}%)"
  echo "Uniform 8-bit model: ${!int8_model_var}"
  echo "Uniform 8-bit accuracy: ${!uniform_acc_var:-Unknown}% (Top-5: ${!uniform_acc5_var:-Unknown}%)"
  echo "Mixed precision strategy: ${!strategy_var}"
  echo "Final mixed precision model: ${!final_model_var}"
  echo "Final accuracy: ${!final_acc_var:-Unknown}% (Top-5: ${!final_acc5_var:-Unknown}%)"
}

# Calculate and print accuracy analysis
print_accuracy_analysis() {
  local stage_name="$1"  # "Stage 1" or "Stage 2"
  local dataset="$2"
  local max_accuracy_drop="$3"

  local prefix
  if [ "$stage_name" = "Stage 1" ]; then
    prefix="SMALL"
  else
    prefix="LARGE"
  fi

  local baseline_acc_var="${prefix}_BASELINE_ACCURACY"
  local uniform_acc_var="${prefix}_UNIFORM_8BIT_ACCURACY"
  local final_acc_var="${prefix}_FINAL_ACCURACY"

  local baseline_accuracy="${!baseline_acc_var}"
  local uniform_8bit_accuracy="${!uniform_acc_var}"
  local final_accuracy="${!final_acc_var}"

  if [ "${baseline_accuracy:-Unknown}" != "Unknown" ]; then
    echo ""
    echo "$stage_name ($dataset):"

    # Calculate accuracy drop from baseline to 8-bit
    if [ "${uniform_8bit_accuracy:-Unknown}" != "Unknown" ]; then
      local uniform_8bit_diff=$(echo "$baseline_accuracy - $uniform_8bit_accuracy" | bc -l)
      echo "├─ 8-bit model accuracy drop from baseline: ${uniform_8bit_diff}%"
    fi

    # Calculate accuracy drop from baseline to mixed precision
    if [ "${final_accuracy:-Unknown}" != "Unknown" ]; then
      local final_diff=$(echo "$baseline_accuracy - $final_accuracy" | bc -l)
      echo "├─ Mixed precision model accuracy drop from baseline: ${final_diff}%"
      if (( $(echo "$final_diff <= $max_accuracy_drop" | bc -l) )); then
        echo "├─ ✅ Met accuracy constraint (max drop: $max_accuracy_drop%)"
      else
        echo "├─ ❌ Exceeded max accuracy drop (got: ${final_diff}%, max: $max_accuracy_drop%)"
      fi
    fi

    # Compare 8-bit vs mixed precision
    if [ "${uniform_8bit_accuracy:-Unknown}" != "Unknown" ] && [ "${final_accuracy:-Unknown}" != "Unknown" ]; then
      local mixed_vs_8bit=$(echo "$final_accuracy - $uniform_8bit_accuracy" | bc -l)
      if (( $(echo "$mixed_vs_8bit >= 0" | bc -l) )); then
        echo "└─ Mixed precision improved accuracy by ${mixed_vs_8bit}% compared to uniform 8-bit"
      else
        local mixed_vs_8bit_abs=$(echo "$mixed_vs_8bit * -1" | bc -l)
        echo "└─ Mixed precision reduced accuracy by ${mixed_vs_8bit_abs}% compared to uniform 8-bit"
      fi
    fi
  fi
}
