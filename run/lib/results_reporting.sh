#!/bin/bash
##############################################################################
# Copyright (C) 2025 Joel Klein                                              #
# All Rights Reserved                                                        #
#                                                                            #
# This work is licensed under the terms described in the LICENSE file        #
# found in the root directory of this source tree.                           #
##############################################################################

# Results reporting functions for CIM-AQ workflows
# This library provides functions for generating comprehensive workflow reports

# Print final workflow summary
print_workflow_summary() {
  local config_file="$1"
  local enable_large_dataset="$2"

  echo ""
  echo ""
  echo "========================================================="
  echo "🎉 TWO-STAGE CIM-AQ WORKFLOW COMPLETE!"
  echo "========================================================="

  echo ""
  echo "📋 CONFIGURATION USED:"
  echo "└─ Config file: $config_file"

  echo ""
  echo "📊 STAGE 1 SUMMARY ($SMALL_DATASET):"
  echo "├─ FP32 fine-tuned model: $SMALL_FP32_MODEL_FILE"
  echo "├─ Baseline accuracy: ${SMALL_BASELINE_ACCURACY:-Unknown}% (Top-5: ${SMALL_BASELINE_ACCURACY5:-Unknown}%)"
  echo "├─ Uniform 8-bit model: $SMALL_INT8_MODEL_FILE"
  echo "├─ Uniform 8-bit accuracy: ${SMALL_UNIFORM_8BIT_ACCURACY:-Unknown}% (Top-5: ${SMALL_UNIFORM_8BIT_ACCURACY5:-Unknown}%)"
  echo "├─ Discovered strategy: $DISCOVERED_STRATEGY_FILE"
  echo "├─ Final mixed precision model: $SMALL_FINAL_MODEL_FILE"
  echo "└─ Final accuracy: ${SMALL_FINAL_ACCURACY:-Unknown}% (Top-5: ${SMALL_FINAL_ACCURACY5:-Unknown}%)"

  if [ "$enable_large_dataset" = "true" ]; then
    echo ""
    echo "📊 STAGE 2 SUMMARY ($LARGE_DATASET):"
    echo "├─ FP32 fine-tuned model: $LARGE_FP32_MODEL_FILE"
    echo "├─ Baseline accuracy: ${LARGE_BASELINE_ACCURACY:-Unknown}% (Top-5: ${LARGE_BASELINE_ACCURACY5:-Unknown}%)"
    echo "├─ Uniform 8-bit model: $LARGE_INT8_MODEL_FILE"
    echo "├─ Uniform 8-bit accuracy: ${LARGE_UNIFORM_8BIT_ACCURACY:-Unknown}% (Top-5: ${LARGE_UNIFORM_8BIT_ACCURACY5:-Unknown}%)"
    echo "├─ Applied strategy: $DISCOVERED_STRATEGY_FILE"
    echo "├─ Final mixed precision model: $LARGE_FINAL_MODEL_FILE"
    echo "└─ Final accuracy: ${LARGE_FINAL_ACCURACY:-Unknown}% (Top-5: ${LARGE_FINAL_ACCURACY5:-Unknown}%)"
  fi
}

# Print comprehensive accuracy analysis
print_comprehensive_accuracy_analysis() {
  local max_accuracy_drop="$1"
  local enable_large_dataset="$2"

  echo ""
  echo "🔍 ACCURACY ANALYSIS:"

  # Include model_evaluation.sh for accuracy analysis functions
  source "$(dirname "${BASH_SOURCE[0]}")/model_evaluation.sh"

  # Stage 1 accuracy analysis
  print_accuracy_analysis "Stage 1" "$SMALL_DATASET" "$max_accuracy_drop"

  # Stage 2 accuracy analysis
  if [ "$enable_large_dataset" = "true" ]; then
    print_accuracy_analysis "Stage 2" "$LARGE_DATASET" "$max_accuracy_drop"
  fi
}

# Print final completion message
print_completion_message() {
  local config_file="$1"

  echo ""
  echo "========================================================="
  echo "✅ CIM-AQ Full Workflow completed successfully!"
  echo "📋 Configuration: $config_file"
  echo "📁 Results saved in respective checkpoint directories"
  echo "========================================================="
}

# Generate complete workflow report
generate_workflow_report() {
  local config_file="$1"
  local max_accuracy_drop="$2"
  local enable_large_dataset="$3"

  print_workflow_summary "$config_file" "$enable_large_dataset"
  print_comprehensive_accuracy_analysis "$max_accuracy_drop" "$enable_large_dataset"
  print_completion_message "$config_file"
}
