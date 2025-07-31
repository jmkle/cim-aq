#!/bin/bash

# CIM-AQ Full Workflow Script
# This script orchestrates a two-stage CIM-AQ workflow using YAML configuration

# Get the directory of the script and the repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Source library functions
source "${SCRIPT_DIR}/lib/workflow_config.sh"
source "${SCRIPT_DIR}/lib/stage_execution.sh"
source "${SCRIPT_DIR}/lib/model_evaluation.sh"
source "${SCRIPT_DIR}/lib/results_reporting.sh"

# Usage information
show_usage() {
  echo "Usage: $0 <config_file>"
  echo ""
  echo "This script runs a two-stage CIM-AQ workflow using YAML configuration:"
  echo ""
  echo "STAGE 1 (Small Dataset - for RL policy discovery):"
  echo "  1. FP32 pretraining (if needed)"
  echo "  2. INT8 pretraining (if needed)"
  echo "  3. RL-based quantization search"
  echo "  4. Mixed precision fine-tuning"
  echo "  5. Final evaluation and comparison"
  echo ""
  echo "STAGE 2 (Large Dataset - using discovered policy):"
  echo "  1. FP32 pretraining (if needed)"
  echo "  2. INT8 pretraining (if needed)"
  echo "  3. Mixed precision fine-tuning (using strategy from Stage 1)"
  echo "  4. Final evaluation and comparison"
  echo ""
  echo "Parameters:"
  echo "  config_file    - Path to YAML configuration file (relative to repo root or absolute path)"
  echo ""
  echo "The configuration file should contain all necessary parameters in YAML format."
  echo "See config_template.yaml for the complete parameter specification."
  echo ""
  echo "Examples:"
  echo "  $0 run/configs/example_config.yaml"
  echo "  $0 /path/to/my_experiment.yaml"
}

# Check arguments
if [ $# -ne 1 ]; then
  show_usage
  exit 1
fi

CONFIG_FILE="$1"

echo "========================================================="
echo "CIM-AQ Full Workflow"
echo "Using configuration file: $CONFIG_FILE"
echo "========================================================="

# Load and validate configuration
echo "Loading configuration..."
if ! load_workflow_config "$CONFIG_FILE" "$REPO_ROOT" "$SCRIPT_DIR"; then
  echo "Failed to load configuration. Exiting."
  exit 1
fi

echo "✅ Configuration loaded successfully"

# Setup workflow variables and paths
setup_workflow_variables "$REPO_ROOT"

# Print configuration summary
print_workflow_config "$CONFIG_FILE" "$REPO_ROOT"

# ========================================
# STAGE 1: SMALL DATASET (RL POLICY DISCOVERY)
# ========================================

echo ""
echo "Starting Stage 1: Small Dataset Policy Discovery..."

if ! execute_stage "Stage 1" "$SMALL_DATASET" "$SMALL_DATASET_ROOT" "$SCRIPT_DIR" "$REPO_ROOT" "false"; then
  echo "❌ Stage 1 execution failed"
  exit 1
fi

# Evaluate Stage 1 models
echo ""
echo "========== Stage 1.5: Evaluation on $SMALL_DATASET =========="
evaluate_stage_models "SMALL" "$SMALL_DATASET" "$SMALL_DATASET_ROOT" "$REPO_ROOT"

# Print Stage 1 results
print_stage_results "Stage 1" "$SMALL_DATASET"

# ========================================
# STAGE 2: LARGE DATASET (APPLY DISCOVERED POLICY)
# ========================================

if [ "$ENABLE_LARGE_DATASET" = "true" ]; then
  echo ""
  echo ""
  echo "Starting Stage 2: Large Dataset Policy Application..."

  if ! execute_stage "Stage 2" "$LARGE_DATASET" "$LARGE_DATASET_ROOT" "$SCRIPT_DIR" "$REPO_ROOT" "true"; then
    echo "❌ Stage 2 execution failed"
    exit 1
  fi

  # Evaluate Stage 2 models
  echo ""
  echo "========== Stage 2.4: Evaluation on $LARGE_DATASET =========="
  evaluate_stage_models "LARGE" "$LARGE_DATASET" "$LARGE_DATASET_ROOT" "$REPO_ROOT"

  # Print Stage 2 results
  print_stage_results "Stage 2" "$LARGE_DATASET"
else
  echo ""
  echo "⏭️  Skipping Stage 2 (Large Dataset) - disabled or same as small dataset"
fi

# ========================================
# FINAL SUMMARY AND COMPARISON
# ========================================

generate_workflow_report "$CONFIG_FILE" "$MAX_ACCURACY_DROP" "$ENABLE_LARGE_DATASET"
