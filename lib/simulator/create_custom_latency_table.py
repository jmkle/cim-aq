#!/usr/bin/env python3
##############################################################################
# Copyright (C) 2025 Joel Klein                                              #
# All Rights Reserved                                                        #
#                                                                            #
# This work is licensed under the terms described in the LICENSE file        #
# found in the root directory of this source tree.                           #
##############################################################################

import sys
from pathlib import Path
from typing import Any

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import yaml

from lib.utils.logger import logger
"""
This script creates a custom latency lookup table for a crossbar-based MVM hardware.
The table estimates latency based on bit-widths of weights and activations and matrix dimensions.

The script takes YAML configuration files for:
1. Hardware parameters
2. Layer dimensions for the model
"""


def parse_int(value: str | int) -> int:
    """
    Parses a string or integer value into an integer.
    If the value is a string containing an expression (e.g., '64*224*224'),
    it evaluates the expression and returns the integer result.

    Args:
        value (str | int): The value to parse, can be a string or an integer

    Returns:
        int: The parsed integer value

    Raises:
        ValueError: If the value cannot be parsed as an integer
    """
    try:
        if isinstance(value, str) and '*' in value:
            return int(eval(value))
        else:
            return int(value)
    except Exception as e:
        raise ValueError(f"Cannot parse {value} as int") from e


def create_crossbar_latency_table(
        model_name: str, max_bit: int, layer_dimensions_yaml: list[dict[str,
                                                                        Any]],
        hardware_config: dict[str, Any]) -> tuple[np.ndarray, Path]:
    """
    Creates a latency lookup table for a crossbar-based MVM hardware for each layer x weight bit x activation bit.
    
    Args:
        model_name (str): Name of the model to use in the output filename
        max_bit (int): Maximum bit-width to sweep over for weights and activations
        layer_dimensions_yaml (list): List of layer dimensions from YAML file
        hardware_config (dict): Dictionary loaded from YAML with hardware parameters

    Returns:
        A 3D numpy array where dimensions are [layer_idx, weight_bits, activation_bits]
        Each entry represents the relative latency for the given configuration.
    """
    # Determine number of layers from the layer dimensions
    num_layers = len(layer_dimensions_yaml)
    logger.info(
        f"Creating latency table for {model_name} with {num_layers} layers")

    # Initialize the lookup table
    latency_table = np.zeros((num_layers, max_bit, max_bit))

    # Read crossbar config
    crossbar_size_m = hardware_config['crossbar']['size']['m']
    crossbar_size_n = hardware_config['crossbar']['size']['n']
    cell_resolution = hardware_config['crossbar']['resolution_weight_bits']
    input_resolution = hardware_config['crossbar']['resolution_input_bits']
    mvm_write_latency = hardware_config['crossbar']['mvm_write_latency']
    mvm_execute_latency = hardware_config['crossbar']['mvm_execute_latency']
    mapping_type = hardware_config['mapper']['mapping_type']

    logger.info(
        f"Using crossbar size: {crossbar_size_m}x{crossbar_size_n}, cell resolution: {cell_resolution} bits"
    )

    # For each layer in the model
    for layer_idx in range(num_layers):
        # Get layer information
        layer_info = layer_dimensions_yaml[layer_idx]

        m = parse_int(layer_info['output_dim'])
        n = parse_int(layer_info['input_dim'])
        mvm_invocations = parse_int(layer_info.get('mvm_invocations', 1))
        repeat_factor = parse_int(layer_info.get('repeat_factor', 1))

        # For each weight bit configuration
        for w_bit in range(1, max_bit + 1):
            # For each activation bit configuration
            for a_bit in range(1, max_bit + 1):
                if mapping_type == 'linear-scaling':
                    num_mvm_writes = np.ceil(m / crossbar_size_m * np.ceil(
                        w_bit / cell_resolution)) * np.ceil(
                            n / crossbar_size_n)
                else:
                    # differential mode mapping -> each weight is mapped to two cells
                    num_mvm_writes = np.ceil(2 * m / crossbar_size_m * np.ceil(
                        w_bit / cell_resolution)) * np.ceil(
                            n / crossbar_size_n)

                # Calculate the number of MVM executes
                num_mvm_executes = mvm_invocations * num_mvm_writes * np.ceil(
                    a_bit / input_resolution)

                # Calculate the total latency for this configuration
                latency = repeat_factor * (
                    num_mvm_writes * mvm_write_latency +
                    num_mvm_executes * mvm_execute_latency)

                # Store in the lookup table (zero-indexed)
                latency_table[layer_idx, w_bit - 1, a_bit - 1] = latency

    # Save the table
    save_dir = Path('lib') / 'simulator' / 'lookup_tables'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{model_name}_batch1_latency_table.npy'
    np.save(save_path, latency_table)

    logger.info(f'Latency table created and saved at {save_path}')
    return latency_table, save_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=
        'Create a custom latency lookup table based on YAML configurations')

    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Name to use for the output file (e.g., vgg16, resnet18)')

    parser.add_argument('--max_bit',
                        type=int,
                        default=8,
                        help='Maximum bit-width to consider')

    parser.add_argument('--layer_dims_yaml',
                        type=str,
                        required=True,
                        help='YAML file path with layer dimensions')

    parser.add_argument('--hardware_config_yaml',
                        type=str,
                        required=True,
                        help='YAML config file for hardware')

    args = parser.parse_args()

    # Load hardware configuration YAML
    try:
        with open(args.hardware_config_yaml, 'r') as f:
            hardware_config = yaml.safe_load(f)
        logger.info(
            f'Loaded hardware configuration from {args.hardware_config_yaml}')
    except Exception as e:
        logger.error(f'Error loading hardware configuration: {e}')
        exit(1)

    # Load layer dimensions YAML
    try:
        with open(args.layer_dims_yaml, 'r') as f:
            dims_yaml = yaml.safe_load(f)
            layer_dimensions = dims_yaml['layer_dimensions']
        logger.info(f'Loaded layer dimensions from {args.layer_dims_yaml}')
        logger.info(f'Number of layers: {len(layer_dimensions)}')
    except Exception as e:
        logger.error(f'Error loading layer dimensions: {e}')
        exit(1)

    latency_table, save_path = create_crossbar_latency_table(
        args.model_name,
        max_bit=args.max_bit,
        layer_dimensions_yaml=layer_dimensions,
        hardware_config=hardware_config)

    logger.info(f'Example latency values for first layer:')
    logger.info(f'W=8bit, A=8bit: {latency_table[0, 7, 7]}')
    logger.info(f'W=4bit, A=4bit: {latency_table[0, 3, 3]}')
    logger.info(f'W=2bit, A=2bit: {latency_table[0, 1, 1]}')
