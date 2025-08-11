#!/usr/bin/env python3
##############################################################################
# Copyright (C) 2025 Joel Klein                                              #
# All Rights Reserved                                                        #
#                                                                            #
# This work is licensed under the terms described in the LICENSE file        #
# found in the root directory of this source tree.                           #
##############################################################################

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.utils.logger import logger


@dataclass
class LayerResult:
    """Data class for storing layer analysis results."""
    layer_idx: int
    weight_bits: int
    activation_bits: int
    cost: float
    writes: float = 0.0
    executes: float = 0.0
    layer_type: str = 'Conv2D'
    mvm_invocations: int = 1
    repeat_factor: int = 1
    matrix_dims: str = 'N/A'
    m_dim_raw: str = 'N/A'
    m_dim_processed: str | int = 'N/A'
    n_dim_raw: str = 'N/A'
    n_dim_processed: str | int = 'N/A'

    @property
    def total_mvms(self) -> int:
        """Calculate total MVM operations."""
        return self.mvm_invocations * self.repeat_factor


@dataclass
class AnalysisResults:
    """Data class for storing complete analysis results."""
    num_layers: int
    strategy_path: str
    lookup_table_path: str
    layer_costs: np.ndarray
    layer_writes: np.ndarray | None
    layer_executes: np.ndarray | None
    layer_details: list[LayerResult]

    @property
    def total_cost(self) -> float:
        """Calculate total cost across all layers."""
        return float(np.sum(self.layer_costs))

    @property
    def avg_cost(self) -> float:
        """Calculate average cost per layer."""
        return float(np.mean(self.layer_costs))

    @property
    def max_cost(self) -> float:
        """Calculate maximum layer cost."""
        return float(np.max(self.layer_costs))

    @property
    def min_cost(self) -> float:
        """Calculate minimum layer cost."""
        return float(np.min(self.layer_costs))

    @property
    def total_writes(self) -> float | None:
        """Calculate total writes if available."""
        return float(np.sum(
            self.layer_writes)) if self.layer_writes is not None else None

    @property
    def total_executes(self) -> float | None:
        """Calculate total executes if available."""
        return float(np.sum(
            self.layer_executes)) if self.layer_executes is not None else None

    @property
    def avg_writes(self) -> float | None:
        """Calculate average writes per layer if available."""
        return float(np.mean(
            self.layer_writes)) if self.layer_writes is not None else None

    @property
    def avg_executes(self) -> float | None:
        """Calculate average executes per layer if available."""
        return float(np.mean(
            self.layer_executes)) if self.layer_executes is not None else None


class QuantizationAnalysisError(Exception):
    """Base exception for quantization analysis errors."""
    pass


class FileLoadError(QuantizationAnalysisError):
    """Exception raised when files cannot be loaded."""
    pass


class ValidationError(QuantizationAnalysisError):
    """Exception raised when validation fails."""
    pass


class ConfigurationLoader:
    """Handles loading and validation of configuration files."""

    @staticmethod
    def load_strategy(strategy_path: str) -> np.ndarray:
        """Load quantization strategy from file.
        
        Args:
            strategy_path: Path to strategy .npy file
            
        Returns:
            Loaded strategy array
            
        Raises:
            FileLoadError: If file cannot be loaded
        """
        try:
            strategy = np.load(strategy_path)
            logger.info(f"✓ Loaded strategy from {strategy_path}")
            logger.info(f"  Strategy shape: {strategy.shape}")
            return strategy
        except Exception as e:
            raise FileLoadError(
                f"Error loading strategy from {strategy_path}: {e}")

    @staticmethod
    def load_lookup_table(lookup_table_path: str) -> np.ndarray:
        """Load lookup table from file.
        
        Args:
            lookup_table_path: Path to lookup table .npy file
            
        Returns:
            Loaded lookup table array
            
        Raises:
            FileLoadError: If file cannot be loaded
        """
        try:
            lookup_table = np.load(lookup_table_path)
            logger.info(f"✓ Loaded lookup table from {lookup_table_path}")
            logger.info(f"  Lookup table shape: {lookup_table.shape}")
            return lookup_table
        except Exception as e:
            raise FileLoadError(
                f"Error loading lookup table from {lookup_table_path}: {e}")

    @staticmethod
    def load_hardware_config(
            hardware_config_path: str) -> dict[str, Any] | None:
        """Load hardware configuration from YAML file.
        
        Args:
            hardware_config_path: Path to hardware config YAML file
            
        Returns:
            Hardware configuration dictionary or None if load fails
        """
        try:
            with open(hardware_config_path, 'r') as f:
                hardware_config = yaml.safe_load(f)
            logger.info(
                f'✓ Loaded hardware configuration from {hardware_config_path}')
            return hardware_config
        except Exception as e:
            logger.warning(
                f'⚠ Warning: Error loading hardware configuration: {e}')
            return None

    @staticmethod
    def load_layer_dimensions(
            layer_dims_path: str) -> list[dict[str, Any]] | None:
        """Load layer dimensions from YAML file.
        
        Args:
            layer_dims_path: Path to layer dimensions YAML file
            
        Returns:
            List of layer dimension dictionaries or None if load fails
        """
        try:
            with open(layer_dims_path, 'r') as f:
                dims_yaml = yaml.safe_load(f)
                layer_dimensions = dims_yaml['layer_dimensions']
            logger.info(f'✓ Loaded layer dimensions from {layer_dims_path}')
            logger.info(
                f'  Found {len(layer_dimensions)} layer dimension entries')
            return layer_dimensions
        except Exception as e:
            logger.warning(f'⚠ Warning: Error loading layer dimensions: {e}')
            return None


class DataValidator:
    """Handles validation of input data."""

    @staticmethod
    def validate_file_paths(strategy_path: str,
                            lookup_table_path: str) -> None:
        """Validate that required files exist.
        
        Args:
            strategy_path: Path to strategy file
            lookup_table_path: Path to lookup table file
            
        Raises:
            FileNotFoundError: If files don't exist
        """
        if not Path(strategy_path).exists():
            raise FileNotFoundError(
                f"Strategy file not found: {strategy_path}")

        if not Path(lookup_table_path).exists():
            raise FileNotFoundError(
                f"Lookup table file not found: {lookup_table_path}")

    @staticmethod
    def validate_strategy_format(strategy: np.ndarray) -> None:
        """Validate strategy array format.
        
        Args:
            strategy: Strategy array to validate
            
        Raises:
            ValidationError: If format is invalid
        """
        if len(strategy.shape) != 2:
            raise ValidationError(
                f"Strategy must be 2D array, got {len(strategy.shape)}D")

        if strategy.shape[1] < 2:
            raise ValidationError(
                f"Strategy format error: Expected at least 2 columns, got {strategy.shape[1]}"
            )

    @staticmethod
    def validate_dimensions(strategy: np.ndarray,
                            lookup_table: np.ndarray) -> None:
        """Validate that strategy and lookup table have compatible dimensions.
        
        Args:
            strategy: Strategy array
            lookup_table: Lookup table array
            
        Raises:
            ValidationError: If dimensions are incompatible
        """
        num_layers_strategy = strategy.shape[0]
        num_layers_table = lookup_table.shape[0]

        if num_layers_strategy != num_layers_table:
            raise ValidationError(
                f"Dimension mismatch: Strategy has {num_layers_strategy} layers, "
                f"lookup table has {num_layers_table} layers")

    @staticmethod
    def validate_bit_widths(weight_bits: int, activation_bits: int,
                            max_weight_bits: int, max_activation_bits: int,
                            layer_idx: int) -> None:
        """Validate bit-width values are within acceptable ranges.
        
        Args:
            weight_bits: Weight bit width
            activation_bits: Activation bit width
            max_weight_bits: Maximum allowed weight bits
            max_activation_bits: Maximum allowed activation bits
            layer_idx: Layer index for error reporting
            
        Raises:
            ValidationError: If bit widths are out of range
        """
        if weight_bits < 1 or weight_bits > max_weight_bits:
            raise ValidationError(
                f"Weight bits {weight_bits} out of range [1, {max_weight_bits}] "
                f"for layer {layer_idx}")

        if activation_bits < 1 or activation_bits > max_activation_bits:
            raise ValidationError(
                f"Activation bits {activation_bits} out of range [1, {max_activation_bits}] "
                f"for layer {layer_idx}")


class CrossbarCalculator:
    """Handles crossbar operation calculations."""

    def __init__(self, hardware_config: dict[str, Any]):
        """Initialize calculator with hardware configuration.
        
        Args:
            hardware_config: Hardware configuration dictionary
        """
        self.hardware_config = hardware_config
        self._extract_hardware_params()

    def _extract_hardware_params(self) -> None:
        """Extract hardware parameters from config."""
        crossbar_params = self.hardware_config['crossbar']
        self.crossbar_size_m = crossbar_params['size']['m']
        self.crossbar_size_n = crossbar_params['size']['n']
        self.cell_resolution = crossbar_params['resolution_weight_bits']
        self.input_resolution = crossbar_params['resolution_input_bits']

        # Get mapping type
        self.mapping_type = self.hardware_config.get('mapper', {}).get(
            'mapping_type', 'differential')

    @staticmethod
    def process_dimension(dim: str | int) -> int:
        """Process dimension value, handling string expressions.
        
        Args:
            dim: Dimension value (can be string with * or int)
            
        Returns:
            Processed dimension as integer
        """
        if isinstance(dim, str) and '*' in dim:
            return eval(dim)
        return int(dim)

    def calculate_operations(self,
                             m: int,
                             n: int,
                             weight_bits: int,
                             activation_bits: int,
                             mvm_invocations: int = 1,
                             repeat_factor: int = 1) -> tuple[float, float]:
        """Calculate number of writes and executes for crossbar operations.
        
        Args:
            m: Matrix dimension M
            n: Matrix dimension N
            weight_bits: Weight bit width
            activation_bits: Activation bit width
            mvm_invocations: Number of MVM invocations
            repeat_factor: Repeat factor for operations
            
        Returns:
            Tuple of (writes, executes)
        """
        # Calculate MVM writes based on mapping type
        if self.mapping_type == 'linear-scaling':
            num_mvm_writes = (
                np.ceil(m / self.crossbar_size_m *
                        np.ceil(weight_bits / self.cell_resolution)) *
                np.ceil(n / self.crossbar_size_n))
        else:
            # differential mode mapping -> each weight is mapped to two cells
            num_mvm_writes = (
                np.ceil(2 * m / self.crossbar_size_m *
                        np.ceil(weight_bits / self.cell_resolution)) *
                np.ceil(n / self.crossbar_size_n))

        # Calculate MVM executes
        num_mvm_executes = mvm_invocations * num_mvm_writes * np.ceil(
            activation_bits / self.input_resolution)

        # Apply repeat factor
        num_mvm_writes *= repeat_factor
        num_mvm_executes *= repeat_factor

        return float(num_mvm_writes), float(num_mvm_executes)


class QuantizationAnalyzer:
    """Main analyzer for quantization strategies using lookup tables and crossbar calculations."""

    def __init__(self, hardware_config: dict[str, Any] | None = None):
        """Initialize the analyzer with optional hardware configuration.
        
        Args:
            hardware_config: Dictionary with hardware parameters
        """
        self.hardware_config = hardware_config
        self.crossbar_calculator = CrossbarCalculator(
            hardware_config) if hardware_config else None

    def analyze_strategy(
        self,
        strategy_path: str,
        lookup_table_path: str,
        layer_dimensions: list[dict[str, Any]] | None = None
    ) -> AnalysisResults:
        """Analyze a quantization strategy and return comprehensive results.
        
        Args:
            strategy_path: Path to strategy file
            lookup_table_path: Path to lookup table file
            layer_dimensions: Optional layer dimension information
            
        Returns:
            AnalysisResults object containing analysis results
            
        Raises:
            FileLoadError: If required files cannot be loaded
            ValidationError: If validation fails
        """
        # Validate file paths
        DataValidator.validate_file_paths(strategy_path, lookup_table_path)

        # Load data
        strategy = ConfigurationLoader.load_strategy(strategy_path)
        lookup_table = ConfigurationLoader.load_lookup_table(lookup_table_path)

        # Validate inputs
        DataValidator.validate_strategy_format(strategy)
        DataValidator.validate_dimensions(strategy, lookup_table)

        # Initialize results
        num_layers = strategy.shape[0]
        layer_costs = np.zeros(num_layers)
        layer_writes = np.zeros(num_layers) if self.hardware_config else None
        layer_executes = np.zeros(num_layers) if self.hardware_config else None
        layer_details = []

        max_weight_bits = lookup_table.shape[1]
        max_activation_bits = lookup_table.shape[2]

        # Analyze each layer
        for layer_idx in range(num_layers):
            layer_result = self._analyze_layer(strategy, lookup_table,
                                               layer_idx, layer_dimensions,
                                               max_weight_bits,
                                               max_activation_bits)

            # Store results
            layer_costs[layer_idx] = layer_result.cost
            layer_details.append(layer_result)

            if layer_writes is not None:
                layer_writes[layer_idx] = layer_result.writes
                layer_executes[layer_idx] = layer_result.executes

        return AnalysisResults(num_layers=num_layers,
                               strategy_path=strategy_path,
                               lookup_table_path=lookup_table_path,
                               layer_costs=layer_costs,
                               layer_writes=layer_writes,
                               layer_executes=layer_executes,
                               layer_details=layer_details)

    def _analyze_layer(self, strategy: np.ndarray, lookup_table: np.ndarray,
                       layer_idx: int,
                       layer_dimensions: list[dict[str, Any]] | None,
                       max_weight_bits: int,
                       max_activation_bits: int) -> LayerResult:
        """Analyze a single layer.
        
        Args:
            strategy: Strategy array
            lookup_table: Lookup table array
            layer_idx: Index of layer to analyze
            layer_dimensions: Optional layer dimension information
            max_weight_bits: Maximum weight bits in lookup table
            max_activation_bits: Maximum activation bits in lookup table
            
        Returns:
            LayerResult object with analysis results
        """
        # Extract bit-widths
        weight_bits = int(strategy[layer_idx, 0])
        activation_bits = int(strategy[layer_idx, 1])

        # Handle special values
        if weight_bits == -1:
            weight_bits = max_weight_bits
        if activation_bits == -1:
            activation_bits = max_activation_bits

        # Validate bit-widths
        DataValidator.validate_bit_widths(weight_bits, activation_bits,
                                          max_weight_bits, max_activation_bits,
                                          layer_idx)

        # Get cost from lookup table and convert to seconds
        cost = lookup_table[layer_idx, weight_bits - 1, activation_bits - 1]
        cost_seconds = float(cost) / 1e6  # Convert microseconds to seconds

        # Initialize layer result with defaults
        layer_result = LayerResult(layer_idx=layer_idx,
                                   weight_bits=weight_bits,
                                   activation_bits=activation_bits,
                                   cost=cost_seconds)

        # Calculate crossbar operations if hardware config is available
        if self.crossbar_calculator and layer_dimensions and layer_idx < len(
                layer_dimensions):
            self._calculate_crossbar_operations(layer_result,
                                                layer_dimensions[layer_idx])

        return layer_result

    def _calculate_crossbar_operations(self, layer_result: LayerResult,
                                       layer_info: dict[str, Any]) -> None:
        """Calculate crossbar operations for a layer.
        
        Args:
            layer_result: LayerResult object to update
            layer_info: Layer information dictionary
        """
        layer_type = layer_info.get('type', 'Conv2D')
        m = layer_info['output_dim']
        n = layer_info['input_dim']
        mvm_invocations = layer_info.get('mvm_invocations', 1)
        repeat_factor = layer_info.get('repeat_factor', 1)

        # Store raw dimensions (with * replaced by x for readability)
        m_raw = str(m).replace('*', ' × ') if isinstance(m, str) else str(m)
        n_raw = str(n).replace('*', ' × ') if isinstance(n, str) else str(n)

        # Process dimensions
        m_val = self.crossbar_calculator.process_dimension(m)
        n_val = self.crossbar_calculator.process_dimension(n)
        mvm_invocations_val = self.crossbar_calculator.process_dimension(
            mvm_invocations)
        repeat_factor_val = self.crossbar_calculator.process_dimension(
            repeat_factor)

        # Calculate operations
        writes, executes = self.crossbar_calculator.calculate_operations(
            m_val, n_val, layer_result.weight_bits,
            layer_result.activation_bits, mvm_invocations_val,
            repeat_factor_val)

        # Update layer result
        layer_result.writes = writes
        layer_result.executes = executes
        layer_result.layer_type = layer_type
        layer_result.mvm_invocations = mvm_invocations_val
        layer_result.repeat_factor = repeat_factor_val
        layer_result.matrix_dims = f"{m_val} × {n_val}"
        layer_result.m_dim_raw = m_raw
        layer_result.m_dim_processed = m_val
        layer_result.n_dim_raw = n_raw
        layer_result.n_dim_processed = n_val


class ResultsManager:
    """Manages display and saving of analysis results."""

    @staticmethod
    def print_results(results: AnalysisResults) -> None:
        """Print comprehensive analysis results to console."""
        logger.info(f"{'='*50}")
        logger.info("QUANTIZATION STRATEGY ANALYSIS RESULTS")
        logger.info(f"{'='*50}")

        # Input files section
        logger.info(f"\nInput Files:")
        logger.info(f"  Strategy: {results.strategy_path}")
        logger.info(f"  Lookup table: {results.lookup_table_path}")

        # Summary statistics
        logger.info(f"\nSummary Statistics:")
        logger.info(f"  Number of layers: {results.num_layers}")
        logger.info(f"  Total cost: {results.total_cost:.7f} seconds")
        logger.info(
            f"  Average cost per layer: {results.avg_cost:.7f} seconds")
        logger.info(f"  Max layer cost: {results.max_cost:.7f} seconds")
        logger.info(f"  Min layer cost: {results.min_cost:.7f} seconds")

        if results.total_writes is not None:
            logger.info(f"  Total writes: {results.total_writes:.0f}")
            logger.info(f"  Total executes: {results.total_executes:.0f}")
            logger.info(
                f"  Average writes per layer: {results.avg_writes:.4f}")
            logger.info(
                f"  Average executes per layer: {results.avg_executes:.4f}")

        # Per-layer details
        logger.info(f"\nPer-Layer Details:")
        for detail in results.layer_details:
            output = (
                f"  Layer {detail.layer_idx:2d}: "
                f"W={detail.weight_bits}bit, A={detail.activation_bits}bit, "
                f"Cost={detail.cost:.7f}s")

            if detail.writes > 0:
                output += f", Writes={detail.writes:8.0f}, Executes={detail.executes:9.0f}"

            logger.info(output)

        # Add detailed MVM information if available
        ResultsManager._print_mvm_details(results)

    @staticmethod
    def _print_mvm_details(results: AnalysisResults) -> None:
        """Print detailed MVM information if available."""
        if not results.layer_details or not hasattr(results.layer_details[0],
                                                    'mvm_invocations'):
            return

        logger.info(f"\nDetailed MVM Information:")
        logger.info(
            f"{'Layer':<5} {'Type':<7} {'Matrix':<12} {'MVMs':<4} {'Repeat':<6} "
            f"{'Total MVMs':<10} {'Description'}")
        logger.info(f"{'-'*80}")

        for detail in results.layer_details:
            description = ResultsManager._generate_layer_description(detail)

            logger.info(
                f"{detail.layer_idx:<5} {detail.layer_type:<7} {detail.matrix_dims:<12} "
                f"{detail.mvm_invocations:<4} {detail.repeat_factor:<6} "
                f"{detail.total_mvms:<10} {description}")

    @staticmethod
    def _generate_layer_description(detail: LayerResult) -> str:
        """Generate description for layer based on type and parameters."""
        if detail.layer_type == 'Conv2D':
            return f"Spatial conv: {detail.mvm_invocations} spatial positions"
        elif detail.layer_type == 'Dense' and detail.mvm_invocations == 1:
            return "Standard FC: 1 MVM"
        elif detail.layer_type == 'Dense' and detail.mvm_invocations > 1:
            return f"Sequential tokens: {detail.mvm_invocations} MVMs"
        elif detail.layer_type == 'MatMul' and detail.repeat_factor > 1:
            return f"Attention: {detail.mvm_invocations} MVMs × {detail.repeat_factor} heads"
        else:
            return f"Custom: {detail.mvm_invocations} × {detail.repeat_factor}"

    @staticmethod
    def print_results_summary(results: AnalysisResults) -> None:
        """Print a summary version of results without per-layer details."""
        logger.info(f"{'='*50}")
        logger.info("QUANTIZATION STRATEGY ANALYSIS SUMMARY")
        logger.info(f"{'='*50}")

        logger.info(f"\nInput Files:")
        logger.info(f"  Strategy: {results.strategy_path}")
        logger.info(f"  Lookup table: {results.lookup_table_path}")

        logger.info(f"\nSummary Statistics:")
        logger.info(f"  Number of layers: {results.num_layers}")
        logger.info(f"  Total cost: {results.total_cost:.7f} seconds")
        logger.info(
            f"  Average cost per layer: {results.avg_cost:.7f} seconds")
        logger.info(f"  Max layer cost: {results.max_cost:.7f} seconds")
        logger.info(f"  Min layer cost: {results.min_cost:.7f} seconds")

        if results.total_writes is not None:
            logger.info(f"  Total writes: {results.total_writes:.0f}")
            logger.info(f"  Total executes: {results.total_executes:.0f}")
            logger.info(
                f"  Average writes per layer: {results.avg_writes:.4f}")
            logger.info(
                f"  Average executes per layer: {results.avg_executes:.4f}")

        # Add layer type breakdown
        ResultsManager._print_layer_type_breakdown(results)

    @staticmethod
    def _print_layer_type_breakdown(results: AnalysisResults) -> None:
        """Print breakdown of layer types."""
        if not results.layer_details or not hasattr(results.layer_details[0],
                                                    'layer_type'):
            return

        layer_types = {}
        for detail in results.layer_details:
            layer_type = detail.layer_type
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1

        logger.info(f"\nLayer Type Breakdown:")
        for layer_type, count in layer_types.items():
            logger.info(f"  {layer_type}: {count} layers")

    @staticmethod
    def save_results(results: AnalysisResults,
                     output_dir: str,
                     hardware_config_path: str | None = None,
                     layer_dims_path: str | None = None) -> None:
        """Save results to directory with text and CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save text file
        ResultsManager._save_text_results(results, output_path,
                                          hardware_config_path,
                                          layer_dims_path)

        # Save CSV file
        ResultsManager._save_csv_results(results, output_path)

        logger.info(f"✓ Results saved to directory: {output_path}")
        logger.info(f"  - Human-readable: {output_path / 'results.txt'}")
        logger.info(f"  - CSV data: {output_path / 'results.csv'}")

    @staticmethod
    def _save_text_results(results: AnalysisResults, output_path: Path,
                           hardware_config_path: str | None,
                           layer_dims_path: str | None) -> None:
        """Save human-readable text results."""
        txt_file = output_path / "results.txt"
        with open(txt_file, 'w') as f:
            f.write("Quantization Strategy Analysis Results\n")
            f.write("=" * 50 + "\n\n")

            # Input files section
            f.write("Input Files:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Strategy file: {results.strategy_path}\n")
            f.write(f"Lookup table: {results.lookup_table_path}\n")
            if hardware_config_path:
                f.write(f"Hardware config: {hardware_config_path}\n")
            if layer_dims_path:
                f.write(f"Layer dimensions: {layer_dims_path}\n")
            f.write("\n")

            # Summary statistics
            f.write("Summary Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Number of layers: {results.num_layers}\n")
            f.write(f"Total cost: {results.total_cost:.7f} seconds\n")
            f.write(
                f"Average cost per layer: {results.avg_cost:.7f} seconds\n")
            f.write(f"Max layer cost: {results.max_cost:.7f} seconds\n")
            f.write(f"Min layer cost: {results.min_cost:.7f} seconds\n")

            if results.total_writes is not None:
                f.write(f"Total writes: {results.total_writes:.0f}\n")
                f.write(f"Total executes: {results.total_executes:.0f}\n")
                f.write(
                    f"Average writes per layer: {results.avg_writes:.4f}\n")
                f.write(
                    f"Average executes per layer: {results.avg_executes:.4f}\n"
                )

            # Layer type breakdown
            ResultsManager._write_layer_breakdown(f, results)

            # Per-layer details table
            ResultsManager._write_layer_table(f, results)

    @staticmethod
    def _write_layer_breakdown(f, results: AnalysisResults) -> None:
        """Write layer type breakdown to file."""
        if not results.layer_details or not hasattr(results.layer_details[0],
                                                    'layer_type'):
            return

        layer_types = {}
        for detail in results.layer_details:
            layer_type = detail.layer_type
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1

        f.write("\nLayer Type Breakdown:\n")
        f.write("-" * 20 + "\n")
        for layer_type, count in layer_types.items():
            f.write(f"{layer_type}: {count} layers\n")

    @staticmethod
    def _write_layer_table(f, results: AnalysisResults) -> None:
        """Write per-layer details table to file."""
        f.write("\nPer-Layer Details:\n")
        f.write("-" * 20 + "\n")

        # Determine table format based on available data
        has_hardware_data = results.total_writes is not None
        has_dimension_data = (results.layer_details and hasattr(
            results.layer_details[0], 'm_dim_raw')
                              and results.layer_details[0].m_dim_raw != 'N/A')

        if has_hardware_data and has_dimension_data:
            # Full table with all columns
            header = (
                f"{'Layer':<5} | {'Type':<7} | {'W-bit':<5} | {'A-bit':<5} | "
                f"{'Cost (s)':<12} | {'M Raw':<12} | {'M Proc':<8} | {'N Raw':<12} | "
                f"{'N Proc':<8} | {'MVMs':<6} | {'Repeat':<6} | {'Writes':<10} | {'Executes':<12}"
            )
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            for detail in results.layer_details:
                row = (
                    f"{detail.layer_idx:<5} | {detail.layer_type:<7} | "
                    f"{detail.weight_bits:<5} | {detail.activation_bits:<5} | "
                    f"{detail.cost:<12.7f} | {detail.m_dim_raw:<12} | "
                    f"{detail.m_dim_processed:<8} | {detail.n_dim_raw:<12} | "
                    f"{detail.n_dim_processed:<8} | {detail.mvm_invocations:<6} | "
                    f"{detail.repeat_factor:<6} | {detail.writes:<10.0f} | "
                    f"{detail.executes:<12.0f}")
                f.write(row + "\n")
        elif has_hardware_data:
            # Table with hardware operations but no matrix dimensions
            header = (
                f"{'Layer':<5} | {'Type':<7} | {'W-bit':<5} | {'A-bit':<5} | "
                f"{'Cost (s)':<12} | {'Writes':<10} | {'Executes':<12}")
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            for detail in results.layer_details:
                row = (
                    f"{detail.layer_idx:<5} | {detail.layer_type:<7} | "
                    f"{detail.weight_bits:<5} | {detail.activation_bits:<5} | "
                    f"{detail.cost:<12.7f} | {detail.writes:<10.0f} | "
                    f"{detail.executes:<12.0f}")
                f.write(row + "\n")
        else:
            # Basic table with only quantization information
            header = (
                f"{'Layer':<5} | {'Type':<7} | {'W-bit':<5} | {'A-bit':<5} | "
                f"{'Cost (s)':<12}")
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            for detail in results.layer_details:
                row = (
                    f"{detail.layer_idx:<5} | {detail.layer_type:<7} | "
                    f"{detail.weight_bits:<5} | {detail.activation_bits:<5} | "
                    f"{detail.cost:<12.7f}")
                f.write(row + "\n")

    @staticmethod
    def _save_csv_results(results: AnalysisResults, output_path: Path) -> None:
        """Save CSV results."""
        csv_file = output_path / "results.csv"

        # Determine CSV columns based on available data
        fieldnames = [
            'layer_idx', 'layer_type', 'weight_bits', 'activation_bits',
            'cost_seconds'
        ]

        if results.total_writes is not None:
            fieldnames.extend([
                'writes', 'executes', 'matrix_dims', 'm_dim_raw',
                'm_dim_processed', 'n_dim_raw', 'n_dim_processed',
                'mvm_invocations', 'repeat_factor', 'total_mvms'
            ])

        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Write layer data
            for detail in results.layer_details:
                row = {
                    'layer_idx': detail.layer_idx,
                    'layer_type': detail.layer_type,
                    'weight_bits': detail.weight_bits,
                    'activation_bits': detail.activation_bits,
                    'cost_seconds': detail.cost
                }

                if results.total_writes is not None:
                    row.update({
                        'writes': detail.writes,
                        'executes': detail.executes,
                        'matrix_dims': detail.matrix_dims,
                        'm_dim_raw': detail.m_dim_raw,
                        'm_dim_processed': detail.m_dim_processed,
                        'n_dim_raw': detail.n_dim_raw,
                        'n_dim_processed': detail.n_dim_processed,
                        'mvm_invocations': detail.mvm_invocations,
                        'repeat_factor': detail.repeat_factor,
                        'total_mvms': detail.total_mvms
                    })

                writer.writerow(row)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Analyze quantization strategies using lookup tables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis with lookup table only
    python get_cost_from_lookup_table.py --strategy best_policy.npy --lookup_table model_table.npy
    
    # Full analysis with hardware calculations
    python get_cost_from_lookup_table.py --strategy best_policy.npy --lookup_table model_table.npy \\
        --hardware_config_yaml hardware.yaml --layer_dims_yaml layer_dims.yaml --save_results results_dir/

    # Summary mode output
    python get_cost_from_lookup_table.py --strategy best_policy.npy --lookup_table model_table.npy --summary
        """)

    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        help=
        'Path to the best_policy.npy file containing the quantization strategy'
    )

    parser.add_argument(
        '--lookup_table',
        type=str,
        required=True,
        help=
        'Path to the lookup table .npy file (e.g., model_batch1_latency_table.npy)'
    )

    parser.add_argument(
        '--hardware_config_yaml',
        type=str,
        help=
        'YAML config file for hardware (optional, enables writes/executes calculation)'
    )

    parser.add_argument(
        '--layer_dims_yaml',
        type=str,
        help=
        'YAML file with layer dimensions (optional, enables writes/executes calculation)'
    )

    parser.add_argument(
        '--save_results',
        type=str,
        help=
        'Directory to save results (creates results.txt and results.csv files)'
    )

    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show only summary statistics (no per-layer details)')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Show detailed MVM information for each layer')

    args = parser.parse_args()

    try:
        # Load configuration files
        hardware_config = None
        layer_dimensions = None

        if args.hardware_config_yaml:
            hardware_config = ConfigurationLoader.load_hardware_config(
                args.hardware_config_yaml)

        if args.layer_dims_yaml:
            layer_dimensions = ConfigurationLoader.load_layer_dimensions(
                args.layer_dims_yaml)

        # Initialize analyzer
        analyzer = QuantizationAnalyzer(hardware_config)

        # Perform analysis
        results = analyzer.analyze_strategy(args.strategy, args.lookup_table,
                                            layer_dimensions)

        # Display results based on verbosity level
        if args.summary:
            ResultsManager.print_results_summary(results)
        else:
            ResultsManager.print_results(results)

        # Save results if requested
        if args.save_results:
            ResultsManager.save_results(results, args.save_results,
                                        args.hardware_config_yaml,
                                        args.layer_dims_yaml)

    except (FileNotFoundError, FileLoadError, ValidationError) as e:
        logger.error(f"✗ {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
