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

# Space management and cleanup utilities for CIM-AQ workflows
# This library provides functions for cleaning up intermediate files safely

# Track directories created in this run to avoid cleaning up other experiments
declare -a CREATED_CHECKPOINT_DIRS

# Register a checkpoint directory as created in this run
register_checkpoint_dir() {
  local dir_path="$1"
  CREATED_CHECKPOINT_DIRS+=("$dir_path")
}

# Clean up intermediate files safely - only in directories created during this run
cleanup_intermediate_files() {
  local stage_name="$1"
  local repo_root="$2"

  # Check if cleanup is enabled
  if [ "${ENABLE_CLEANUP:-false}" != "true" ]; then
    echo "🔧 Space management disabled - skipping cleanup"
    return 0
  fi

  # Check cleanup frequency
  case "${CLEANUP_FREQUENCY:-end}" in
    "step")
      echo "🧹 Cleaning up intermediate files after $stage_name (frequency: step)..."
      ;;
    "stage")
      if [ "$stage_name" = "Final" ]; then
        echo "🧹 Final cleanup (frequency: stage)..."
      else
        echo "🧹 Cleaning up intermediate files after $stage_name (frequency: stage)..."
      fi
      ;;
    "end")
      if [ "$stage_name" != "Final" ]; then
        echo "🔧 Skipping cleanup after $stage_name (frequency: end)"
        return 0
      else
        echo "🧹 Final cleanup (frequency: end)..."
      fi
      ;;
    *)
      echo "⚠️  Unknown cleanup_frequency: ${CLEANUP_FREQUENCY}, skipping cleanup"
      return 0
      ;;
  esac

  echo "   Cleanup targets: ${CLEANUP_TARGETS:-[\"checkpoints\"]}"

  # Parse cleanup targets (handle both string and array formats)
  local targets="${CLEANUP_TARGETS:-[\"checkpoints\"]}"

  # Clean based on targets - only in directories we created
  if [[ "$targets" == *"checkpoints"* ]] || [[ "$targets" == *"all"* ]]; then
    echo "   → Cleaning checkpoint files (only in directories created this run)..."
    for dir in "${CREATED_CHECKPOINT_DIRS[@]}"; do
      if [ -d "$dir" ]; then
        find "$dir" -name "checkpoint.pth.tar" -type f -delete 2>/dev/null || true
      fi
    done
  fi

  if [[ "$targets" == *"logs"* ]] || [[ "$targets" == *"all"* ]]; then
    echo "   → Cleaning log directories (only in directories created this run)..."
    for dir in "${CREATED_CHECKPOINT_DIRS[@]}"; do
      if [ -d "$dir" ]; then
        find "$dir" -name "logs" -type d -exec rm -rf {} + 2>/dev/null || true
      fi
    done
  fi

  if [[ "$targets" == *"cache"* ]] || [[ "$targets" == *"all"* ]]; then
    echo "   → Cleaning Python cache files..."
    # Python cache cleanup is safe globally as these are regenerated
    find "${repo_root}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find "${repo_root}" -name "*.pyc" -type f -delete 2>/dev/null || true
  fi

  # Show remaining disk usage for directories we created
  echo "   📊 Checkpoint directory sizes after cleanup:"
  for dir in "${CREATED_CHECKPOINT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
      du -sh "$dir" 2>/dev/null || true
    fi
  done

  if [ ${#CREATED_CHECKPOINT_DIRS[@]} -eq 0 ]; then
    echo "   (No checkpoint directories tracked for this run)"
  fi
}
