#!/usr/bin/env python3
##############################################################################
# Copyright (C) 2025 Joel Klein                                              #
# All Rights Reserved                                                        #
#                                                                            #
# This work is licensed under the terms described in the LICENSE file        #
# found in the root directory of this source tree.                           #
##############################################################################
"""
YAML to Shell Variable Parser
Reads a YAML configuration file and outputs shell variable assignments.
"""

import argparse
import sys
from pathlib import Path

import yaml


def parse_config(config_path: Path, repo_root: Path) -> None:
    """Parse YAML config file and return shell variable assignments."""
    if not config_path.exists():
        print(f"Error: Configuration file '{config_path}' not found",
              file=sys.stderr)
        sys.exit(1)

    try:
        with config_path.open('r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML configuration: {e}",
              file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to read configuration file: {e}",
              file=sys.stderr)
        sys.exit(1)

    # Helper function to resolve paths (relative to repo_root or absolute)
    def resolve_path(path_str: str) -> str:
        path = Path(path_str)
        if path.is_absolute():
            return str(path)
        else:
            abs_path = (repo_root / path).absolute()
            return str(abs_path)

    # Extract values with defaults matching the original script
    try:
        # Model configuration
        quant_model = config.get('models', {}).get('quant_model', 'qvgg16')
        fp32_model = config.get('models', {}).get('fp32_model', 'custom_vgg16')

        # Dataset configuration
        datasets = config.get('datasets', {})
        small_dataset = datasets.get('small', {}).get('name', 'imagenet100')
        small_dataset_root = resolve_path(
            datasets.get('small', {}).get('root', 'data/imagenet100'))
        large_dataset = datasets.get('large', {}).get('name', 'imagenet')
        large_dataset_root = resolve_path(
            datasets.get('large', {}).get('root', 'data/imagenet'))
        enable_large_dataset = str(
            datasets.get('large', {}).get('enabled', True)).lower()

        # Quantization configuration
        quant_config = config.get('quantization', {})
        max_accuracy_drop = str(quant_config.get('max_accuracy_drop', 1.0))
        min_bit = str(quant_config.get('min_bit', 2))
        max_bit = str(quant_config.get('max_bit', 8))
        force_first_last_layer = str(
            quant_config.get('force_first_last_layer', True)).lower()
        consider_cell_resolution = str(
            quant_config.get('consider_cell_resolution', False)).lower()

        # Training configuration
        training_config = config.get('training', {})

        # Learning rates - dataset-specific
        lr_config = training_config.get('learning_rates', {})

        # Small dataset learning rates
        small_lr_config = lr_config.get('small_dataset', {})
        small_fp32_lr = str(small_lr_config.get('fp32_pretraining', 0.0005))
        small_int8_lr = str(small_lr_config.get('int8_pretraining', 0.001))
        small_mp_lr = str(
            small_lr_config.get('mixed_precision_finetuning', 0.0005))
        small_rl_finetune_lr = str(small_lr_config.get('rl_finetune', 0.001))

        # Large dataset learning rates
        large_lr_config = lr_config.get('large_dataset', {})
        large_fp32_lr = str(large_lr_config.get('fp32_pretraining', 0.0005))
        large_int8_lr = str(large_lr_config.get('int8_pretraining', 0.001))
        large_mp_lr = str(
            large_lr_config.get('mixed_precision_finetuning', 0.0005))

        train_episodes = str(
            training_config.get('rl', {}).get('train_episodes', 600))

        small_training = training_config.get('small_dataset', {})
        small_fp32_epochs = str(small_training.get('fp32_epochs', 30))
        small_8bit_epochs = str(small_training.get('int8_epochs', 30))
        small_search_finetune_epochs = str(
            small_training.get('search_finetune_epochs', 3))
        small_finetune_epochs = str(
            small_training.get('final_finetune_epochs', 30))

        large_training = training_config.get('large_dataset', {})
        large_fp32_epochs = str(large_training.get('fp32_epochs', 30))
        large_8bit_epochs = str(large_training.get('int8_epochs', 30))
        large_finetune_epochs = str(
            large_training.get('final_finetune_epochs', 30))

        # Output configuration
        output_config = config.get('output', {})
        output_prefix = output_config.get('prefix', 'per-tensor_no_constraint')

        # Device configuration
        device_config = config.get('device', {})
        gpu_id = device_config.get('gpu_id', '1')

        # Logging configuration
        logging_config = config.get('logging', {})
        wandb_config = logging_config.get('wandb', {})
        wandb_enable = str(wandb_config.get('enable', False)).lower()
        wandb_project = wandb_config.get('project', 'cim-aq-quantization')

        # DataLoader configuration
        dataloader_config = config.get('dataloader', {})
        batch_size = str(dataloader_config.get('batch_size', 256))
        num_workers = str(dataloader_config.get('num_workers', 32))

        # Space management configuration
        space_config = config.get('space_management', {})
        enable_cleanup = str(space_config.get('enable_cleanup', False)).lower()
        cleanup_frequency = space_config.get('cleanup_frequency', 'end')

        # Handle cleanup_targets as either string or list
        cleanup_targets = space_config.get('cleanup_targets', 'checkpoints')
        if isinstance(cleanup_targets, list):
            cleanup_targets_str = ','.join(cleanup_targets)
        else:
            cleanup_targets_str = str(cleanup_targets)

    except Exception as e:
        print(f"Error: Failed to extract configuration values: {e}")
        sys.exit(1)

    # Output shell variable assignments
    print(f'QUANT_MODEL="{quant_model}"')
    print(f'FP32_MODEL="{fp32_model}"')
    print(f'SMALL_DATASET="{small_dataset}"')
    print(f'SMALL_DATASET_ROOT="{small_dataset_root}"')
    print(f'LARGE_DATASET="{large_dataset}"')
    print(f'LARGE_DATASET_ROOT="{large_dataset_root}"')
    print(f'MAX_ACCURACY_DROP="{max_accuracy_drop}"')
    print(f'MIN_BIT="{min_bit}"')
    print(f'MAX_BIT="{max_bit}"')
    print(f'TRAIN_EPISODES="{train_episodes}"')
    print(f'SMALL_FINETUNE_EPOCHS="{small_finetune_epochs}"')
    print(f'SMALL_SEARCH_FINETUNE_EPOCHS="{small_search_finetune_epochs}"')
    print(f'SMALL_8BIT_EPOCHS="{small_8bit_epochs}"')
    print(f'SMALL_FP32_EPOCHS="{small_fp32_epochs}"')
    print(f'LARGE_FINETUNE_EPOCHS="{large_finetune_epochs}"')
    print(f'LARGE_8BIT_EPOCHS="{large_8bit_epochs}"')
    print(f'LARGE_FP32_EPOCHS="{large_fp32_epochs}"')
    print(f'FORCE_FIRST_LAST_LAYER="{force_first_last_layer}"')
    print(f'CONSIDER_CELL_RESOLUTION="{consider_cell_resolution}"')
    print(f'ENABLE_LARGE_DATASET="{enable_large_dataset}"')
    print(f'OUTPUT_PREFIX="{output_prefix}"')
    print(f'GPU_ID="{gpu_id}"')
    print(f'WANDB_ENABLE="{wandb_enable}"')
    print(f'WANDB_PROJECT="{wandb_project}"')
    # Dataset-specific learning rate variables
    print(f'SMALL_FP32_LEARNING_RATE="{small_fp32_lr}"')
    print(f'SMALL_INT8_LEARNING_RATE="{small_int8_lr}"')
    print(f'SMALL_MP_LEARNING_RATE="{small_mp_lr}"')
    print(f'SMALL_RL_FINETUNE_LEARNING_RATE="{small_rl_finetune_lr}"')
    print(f'LARGE_FP32_LEARNING_RATE="{large_fp32_lr}"')
    print(f'LARGE_INT8_LEARNING_RATE="{large_int8_lr}"')
    print(f'LARGE_MP_LEARNING_RATE="{large_mp_lr}"')
    # DataLoader configuration variables
    print(f'BATCH_SIZE="{batch_size}"')
    print(f'NUM_WORKERS="{num_workers}"')
    # Space management configuration variables
    print(f'ENABLE_CLEANUP="{enable_cleanup}"')
    print(f'CLEANUP_FREQUENCY="{cleanup_frequency}"')
    print(f'CLEANUP_TARGETS="{cleanup_targets_str}"')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=
        'Parse YAML configuration file and output shell variable assignments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run/configs/config.yaml
  %(prog)s run/configs/experiment1.yaml
  %(prog)s run/configs/config.yaml --repo-root /custom/path/to/repo
        """)

    parser.add_argument(
        'config_path',
        help=
        'Path to YAML configuration file (absolute or relative to repo root)',
        type=Path)
    parser.add_argument('--repo-root',
                        default=Path(__file__).parent.parent.absolute(),
                        type=Path,
                        help='Repository root directory')

    args = parser.parse_args()

    config_path = args.config_path
    repo_root = args.repo_root.absolute()
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    parse_config(config_path=config_path, repo_root=repo_root)


if __name__ == "__main__":
    main()
