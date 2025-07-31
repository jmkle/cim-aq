import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.utils.logger import logger


class QuantizationAnalyzer:
    """
    Analyzer for quantization strategies using lookup tables and crossbar calculations.
    """

    def __init__(self, hardware_config: dict | None = None):
        """
        Initialize the analyzer with optional hardware configuration.
        
        Args:
            hardware_config: Dictionary with hardware parameters
        """
        self.hardware_config = hardware_config
        self._extract_hardware_params()

    def _extract_hardware_params(self):
        """Extract hardware parameters from config."""
        if self.hardware_config:
            crossbar_params = self.hardware_config['crossbar']
            self.crossbar_size_m = crossbar_params['size']['m']
            self.crossbar_size_n = crossbar_params['size']['n']
            self.cell_resolution = crossbar_params['resolution_weight_bits']
            self.input_resolution = crossbar_params['resolution_input_bits']

    def load_strategy(self, strategy_path: str) -> np.ndarray:
        """Load quantization strategy from file."""
        try:
            strategy = np.load(strategy_path)
            logger.info(f"✓ Loaded strategy from {strategy_path}")
            logger.info(f"  Strategy shape: {strategy.shape}")
            return strategy
        except Exception as e:
            raise FileNotFoundError(
                f"Error loading strategy from {strategy_path}: {e}")

    def load_lookup_table(self, lookup_table_path: str) -> np.ndarray:
        """Load lookup table from file."""
        try:
            lookup_table = np.load(lookup_table_path)
            logger.info(f"✓ Loaded lookup table from {lookup_table_path}")
            logger.info(f"  Lookup table shape: {lookup_table.shape}")
            return lookup_table
        except Exception as e:
            raise FileNotFoundError(
                f"Error loading lookup table from {lookup_table_path}: {e}")

    def validate_strategy_format(self, strategy: np.ndarray) -> None:
        """Validate strategy array format."""
        if strategy.shape[1] < 2:
            raise ValueError(
                f"Strategy format error: Expected 2 columns, got {strategy.shape[1]}"
            )

    def validate_dimensions(self, strategy: np.ndarray,
                            lookup_table: np.ndarray) -> None:
        """Validate that strategy and lookup table have compatible dimensions."""
        num_layers_strategy = strategy.shape[0]
        num_layers_table = lookup_table.shape[0]

        if num_layers_strategy != num_layers_table:
            raise ValueError(
                f"Dimension mismatch: Strategy has {num_layers_strategy} layers, "
                f"lookup table has {num_layers_table} layers")

    def validate_bit_widths(self, weight_bits: int, activation_bits: int,
                            max_weight_bits: int, max_activation_bits: int,
                            layer_idx: int) -> None:
        """Validate bit-width values are within acceptable ranges."""
        if weight_bits < 1 or weight_bits > max_weight_bits:
            raise ValueError(
                f"Weight bits {weight_bits} out of range [1, {max_weight_bits}] "
                f"for layer {layer_idx}")

        if activation_bits < 1 or activation_bits > max_activation_bits:
            raise ValueError(
                f"Activation bits {activation_bits} out of range [1, {max_activation_bits}] "
                f"for layer {layer_idx}")

    def process_dimension(self, dim: Any) -> int:
        """Process dimension value, handling string expressions."""
        if isinstance(dim, str) and '*' in dim:
            return eval(dim)
        return int(dim)

    def calculate_crossbar_operations(
            self,
            m: int,
            n: int,
            weight_bits: int,
            activation_bits: int,
            mvm_invocations: int = 1,
            repeat_factor: int = 1) -> tuple[float, float]:
        """Calculate number of writes and executes for crossbar operations."""
        if not self.hardware_config:
            return 0.0, 0.0

        # Calculate MVM writes
        num_mvm_writes = (
            np.ceil(2 * m / self.crossbar_size_m *
                    np.ceil(weight_bits / self.cell_resolution)) *
            np.ceil(2 * n / self.crossbar_size_n *
                    np.ceil(weight_bits / self.cell_resolution)))

        # Calculate MVM executes using mvm_invocations
        num_mvm_executes = mvm_invocations * num_mvm_writes * np.ceil(
            activation_bits / self.input_resolution)

        # Apply repeat factor to both writes and executes
        num_mvm_writes *= repeat_factor
        num_mvm_executes *= repeat_factor

        return float(num_mvm_writes), float(num_mvm_executes)

    def analyze_strategy(
            self,
            strategy_path: str,
            lookup_table_path: str,
            layer_dimensions: list | None = None) -> dict[str, Any]:
        """
        Analyze a quantization strategy and return comprehensive results.
        
        Returns:
            Dictionary containing analysis results
        
        Raises:
            FileNotFoundError: If required files cannot be loaded
            ValueError: If validation fails
        """
        # Load data
        strategy = self.load_strategy(strategy_path)
        lookup_table = self.load_lookup_table(lookup_table_path)

        # Validate inputs
        self.validate_strategy_format(strategy)
        self.validate_dimensions(strategy, lookup_table)

        # Initialize results
        num_layers = strategy.shape[0]
        results = {
            'num_layers': num_layers,
            'strategy_path': strategy_path,
            'lookup_table_path': lookup_table_path,
            'layer_costs': np.zeros(num_layers),
            'layer_writes':
            np.zeros(num_layers) if self.hardware_config else None,
            'layer_executes':
            np.zeros(num_layers) if self.hardware_config else None,
            'layer_details': []
        }

        max_weight_bits = lookup_table.shape[1]
        max_activation_bits = lookup_table.shape[2]

        # Analyze each layer
        for layer_idx in range(num_layers):
            layer_result = self._analyze_layer(strategy, lookup_table,
                                               layer_idx, layer_dimensions,
                                               max_weight_bits,
                                               max_activation_bits)

            # Store results
            results['layer_costs'][layer_idx] = layer_result['cost']
            results['layer_details'].append(layer_result)

            if results['layer_writes'] is not None:
                results['layer_writes'][layer_idx] = layer_result['writes']
                results['layer_executes'][layer_idx] = layer_result['executes']

        # Calculate summary statistics
        results['total_cost'] = np.sum(results['layer_costs'])
        results['avg_cost'] = np.mean(results['layer_costs'])
        results['max_cost'] = np.max(results['layer_costs'])
        results['min_cost'] = np.min(results['layer_costs'])

        if results['layer_writes'] is not None:
            results['total_writes'] = np.sum(results['layer_writes'])
            results['total_executes'] = np.sum(results['layer_executes'])
            results['avg_writes'] = np.mean(results['layer_writes'])
            results['avg_executes'] = np.mean(results['layer_executes'])

        return results

    def _analyze_layer(self, strategy: np.ndarray, lookup_table: np.ndarray,
                       layer_idx: int, layer_dimensions: list | None,
                       max_weight_bits: int, max_activation_bits: int) -> dict:
        """Analyze a single layer."""
        # Extract bit-widths
        weight_bits = int(strategy[layer_idx, 0])
        activation_bits = int(strategy[layer_idx, 1])

        # Handle special values
        if weight_bits == -1:
            weight_bits = max_weight_bits
        if activation_bits == -1:
            activation_bits = max_activation_bits

        # Validate bit-widths
        self.validate_bit_widths(weight_bits, activation_bits, max_weight_bits,
                                 max_activation_bits, layer_idx)

        # Get cost from lookup table and convert to seconds
        cost = lookup_table[layer_idx, weight_bits - 1, activation_bits - 1]
        cost_seconds = float(cost) / 1e6  # Convert microseconds to seconds

        # Initialize layer result
        layer_result = {
            'layer_idx': layer_idx,
            'weight_bits': weight_bits,
            'activation_bits': activation_bits,
            'cost': cost_seconds,
            'writes': 0.0,
            'executes': 0.0,
            'layer_type': 'Conv2D',
            'input_dim': 'N/A',
            'output_dim': 'N/A'
        }

        # Calculate crossbar operations
        if self.hardware_config and layer_dimensions:
            layer_info = layer_dimensions[layer_idx]

            layer_type = layer_info.get('type', 'Conv2D')
            m = layer_info['output_dim']
            n = layer_info['input_dim']
            mvm_invocations = layer_info.get('mvm_invocations', 1)
            repeat_factor = layer_info.get('repeat_factor', 1)

            m_val = self.process_dimension(m)
            n_val = self.process_dimension(n)
            mvm_invocations_val = self.process_dimension(mvm_invocations)
            repeat_factor_val = self.process_dimension(repeat_factor)

            writes, executes = self.calculate_crossbar_operations(
                m_val, n_val, weight_bits, activation_bits,
                mvm_invocations_val, repeat_factor_val)

            layer_result['writes'] = writes
            layer_result['executes'] = executes
            layer_result['layer_type'] = layer_type

            # Extract formatted dimensions
            input_dim_str, output_dim = self._extract_layer_info(layer_info)

            layer_result['input_dim'] = input_dim_str
            layer_result['output_dim'] = output_dim

        return layer_result

    def _extract_layer_info(self, layer_info: dict) -> tuple[str, int]:
        """Extract formatted dimensions from new YAML format."""
        input_dim = layer_info['input_dim']
        output_dim = self.process_dimension(layer_info['output_dim'])

        # Format input dimension string
        if isinstance(input_dim, str) and '*' in input_dim:
            # Parse input dimension string (e.g., "3*224*224" or "512*7*7")
            parts = input_dim.split('*')
            if len(parts) == 3:
                channels, height, width = parts
                input_dim_str = f"\\numproduct[product-symbol = \\ensuremath{{\\times}}]{{{channels} x {height} x {width}}}"
            else:
                input_dim_str = f"\\numproduct[product-symbol = \\ensuremath{{\\times}}]{{{input_dim}}}"
        else:
            # Simple number
            input_val = self.process_dimension(input_dim)
            input_dim_str = f"\\num{{{input_val}}}"

        return input_dim_str, output_dim


class ResultsManager:
    """Manages saving and displaying results."""

    @staticmethod
    def print_results(results: dict[str, Any]):
        """Print analysis results to console."""
        logger.info(f"\n{'='*50}")
        logger.info("QUANTIZATION STRATEGY ANALYSIS RESULTS")
        logger.info(f"{'='*50}")

        logger.info(f"\nInput Files:")
        logger.info(f"  Strategy: {results['strategy_path']}")
        logger.info(f"  Lookup table: {results['lookup_table_path']}")

        logger.info(f"\nSummary Statistics:")
        logger.info(f"  Number of layers: {results['num_layers']}")
        logger.info(f"  Total cost: {results['total_cost']:.7f} seconds")
        logger.info(
            f"  Average cost per layer: {results['avg_cost']:.7f} seconds")
        logger.info(f"  Max layer cost: {results['max_cost']:.7f} seconds")
        logger.info(f"  Min layer cost: {results['min_cost']:.7f} seconds")

        if 'total_writes' in results:
            logger.info(f"  Total writes: {results['total_writes']:.0f}")
            logger.info(f"  Total executes: {results['total_executes']:.0f}")
            logger.info(
                f"  Average writes per layer: {results['avg_writes']:.4f}")
            logger.info(
                f"  Average executes per layer: {results['avg_executes']:.4f}")

        logger.info(f"\nPer-Layer Details:")
        for detail in results['layer_details']:
            output = (
                f"  Layer {detail['layer_idx']:2d}: "
                f"W={detail['weight_bits']}bit, A={detail['activation_bits']}bit, "
                f"Cost={detail['cost']:.7f}s")

            if detail['writes'] > 0:
                output += f", Writes={detail['writes']:8.0f}, Executes={detail['executes']:9.0f}"

            logger.info(output)

    @staticmethod
    def save_results(results: dict[str, Any],
                     output_path: str,
                     hardware_config_path: str | None = None,
                     layer_dims_path: str | None = None):
        """Save results to a text file."""
        # Ensure .txt extension using pathlib
        output_file = Path(output_path)
        if output_file.suffix != '.txt':
            output_file = output_file.with_suffix('.txt')

        with open(output_file, 'w') as f:
            f.write("Quantization Strategy Analysis Results\n")
            f.write("=" * 50 + "\n\n")

            # Input files section
            f.write("Input Files:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Strategy file: {results['strategy_path']}\n")
            f.write(f"Lookup table: {results['lookup_table_path']}\n")
            if hardware_config_path:
                f.write(f"Hardware config: {hardware_config_path}\n")
            if layer_dims_path:
                f.write(f"Layer dimensions: {layer_dims_path}\n")
            f.write("\n")

            # Summary statistics
            f.write("Summary Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Number of layers: {results['num_layers']}\n")
            f.write(f"Total cost: {results['total_cost']:.7f} seconds\n")
            f.write(
                f"Average cost per layer: {results['avg_cost']:.7f} seconds\n")
            f.write(f"Max layer cost: {results['max_cost']:.7f} seconds\n")
            f.write(f"Min layer cost: {results['min_cost']:.7f} seconds\n")

            if 'total_writes' in results:
                f.write(f"Total writes: {results['total_writes']:.0f}\n")
                f.write(f"Total executes: {results['total_executes']:.0f}\n")
                f.write(
                    f"Average writes per layer: {results['avg_writes']:.4f}\n")
                f.write(
                    f"Average executes per layer: {results['avg_executes']:.4f}\n"
                )

            # Per-layer details
            f.write("\nPer-Layer Details:\n")
            f.write("-" * 20 + "\n")

            # Header
            header = "Layer | W-bits | A-bits | Latency (in s)"
            if 'total_writes' in results:
                header += " |   Writes |  Executes"
            f.write(header + "\n")

            # Separator
            separator = "------|--------|--------|----------------"
            if 'total_writes' in results:
                separator += "|----------|----------"
            f.write(separator + "\n")

            # Data rows
            for detail in results['layer_details']:
                row = (f"{detail['layer_idx']:5d} | "
                       f"{detail['weight_bits']:6d} | "
                       f"{detail['activation_bits']:6d} | "
                       f"{detail['cost']:14.7f}")

                if 'total_writes' in results:
                    row += (f" | {detail['writes']:8.0f} | "
                            f"{detail['executes']:9.0f}")

                f.write(row + "\n")

        logger.info(f"✓ Results saved to {output_file}")

    @staticmethod
    def save_latex_table(results: dict[str, Any],
                         output_path: str,
                         caption: str = "Quantization strategy statistics",
                         label: str = "tab:quant_stats"):
        """Save results as a LaTeX table."""
        # Ensure .tex extension using pathlib
        output_file = Path(output_path)
        if output_file.suffix != '.tex':
            output_file = output_file.with_suffix('.tex')

        with open(output_file, 'w') as f:
            f.write("\\begin{table}\n")
            f.write("  \\centering\n")
            f.write("  \\small\n")
            f.write(f"  \\caption{{{caption}}}\n")
            f.write(f"  \\label{{{label}}}\n")

            # Determine column format based on available data
            has_crossbar = 'total_writes' in results
            if has_crossbar:
                col_format = "c|c|cc|cc|cc|c"
            else:
                col_format = "c|c|cc|c"

            f.write(f"  \\begin{{tabular}}{{{col_format}}}\n")

            # Table header
            if has_crossbar:
                f.write(
                    "    \\multirow{2}{*}{\\textbf{Layer}} & \\multirow{2}{*}{\\textbf{Type}} & \\multicolumn{2}{c|}{\\textbf{Bit widths}} & \\multicolumn{2}{c|}{\\textbf{Dimensions}} & \\multicolumn{2}{c|}{\\textbf{Number of}} & \\multirow{2}{*}{\\textbf{Latency (in \\unit{\\second})}} \\\\\n"
                )
                f.write("    \\cline{3-8}\n")
                f.write(
                    "    & & \\textbf{Weight} & \\textbf{Activation} & \\textbf{Input} & \\textbf{Output} & \\textbf{Writes} & \\textbf{Executes} & \\\\\n"
                )
            else:
                f.write(
                    "    \\textbf{Layer} & \\textbf{Type} & \\textbf{Weight} & \\textbf{Activation} & \\textbf{Latency (in \\unit{\\second})} \\\\\n"
                )

            f.write("    \\hline\n")

            # Table rows
            for detail in results['layer_details']:
                layer_num = detail['layer_idx'] + 1
                layer_type = detail.get('layer_type', 'Conv2D')
                weight_bits = detail['weight_bits']
                activation_bits = detail['activation_bits']
                cost = detail['cost']

                f.write("    \\hline\n")

                if has_crossbar:
                    input_dim = detail.get('input_dim', 'N/A')
                    output_dim = detail.get('output_dim', 'N/A')
                    writes = int(detail['writes'])
                    executes = int(detail['executes'])

                    f.write(
                        f"    \\num{{{layer_num}}} & {layer_type} & \\num{{{weight_bits}}} & \\num{{{activation_bits}}} & {input_dim} & \\num{{{output_dim}}} & \\num{{{writes}}} & \\num{{{executes}}} & \\num[round-mode=places,round-precision=3]{{{cost:.7f}}} \\\\\n"
                    )
                else:
                    f.write(
                        f"    \\num{{{layer_num}}} & {layer_type} & \\num{{{weight_bits}}} & \\num{{{activation_bits}}} & \\num[round-mode=places,round-precision=3]{{{cost:.7f}}} \\\\\n"
                    )

            f.write("  \\end{tabular}\n")
            f.write("\\end{table}\n")

        logger.info(f"✓ LaTeX table saved to {output_file}")


def load_config_files(
        hardware_config_path: str | None,
        layer_dims_path: str | None) -> tuple[dict | None, list | None]:
    """Load configuration files."""
    hardware_config = None
    layer_dimensions = None

    if hardware_config_path:
        try:
            with open(hardware_config_path, 'r') as f:
                hardware_config = yaml.safe_load(f)
            logger.info(
                f'✓ Loaded hardware configuration from {hardware_config_path}')
        except Exception as e:
            logger.info(
                f'⚠ Warning: Error loading hardware configuration: {e}')

    if layer_dims_path:
        try:
            with open(layer_dims_path, 'r') as f:
                dims_yaml = yaml.safe_load(f)
                layer_dimensions = dims_yaml['layer_dimensions']
            logger.info(f'✓ Loaded layer dimensions from {layer_dims_path}')
            logger.info(
                f'  Found {len(layer_dimensions)} layer dimension entries')
        except Exception as e:
            logger.info(f'⚠ Warning: Error loading layer dimensions: {e}')

    return hardware_config, layer_dimensions


def validate_file_paths(strategy_path: str, lookup_table_path: str) -> None:
    """Validate that required files exist."""
    if not Path(strategy_path).exists():
        raise FileNotFoundError(f"Strategy file not found: {strategy_path}")

    if not Path(lookup_table_path).exists():
        raise FileNotFoundError(
            f"Lookup table file not found: {lookup_table_path}")


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
        --hardware_config_yaml hardware.yaml --layer_dims_yaml layer_dims.yaml --save_results results.txt

    # Export LaTeX table
    python get_cost_from_lookup_table.py --strategy best_policy.npy --lookup_table model_table.npy \\
        --export_latex table.tex --latex_caption "VGG-16 quantization statistics" --latex_label "tab:vgg16_stats"
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
        help='Path to save results as human-readable text file')

    parser.add_argument('--export_latex',
                        type=str,
                        help='Path to save results as LaTeX table (.tex file)')

    parser.add_argument(
        '--latex_caption',
        type=str,
        default='Quantization strategy statistics',
        help=
        'Caption for LaTeX table (default: "Quantization strategy statistics")'
    )

    parser.add_argument(
        '--latex_label',
        type=str,
        default='tab:quant_stats',
        help='Label for LaTeX table (default: "tab:quant_stats")')

    args = parser.parse_args()

    try:
        # Validate required files
        validate_file_paths(args.strategy, args.lookup_table)

        # Load configuration files
        hardware_config, layer_dimensions = load_config_files(
            args.hardware_config_yaml, args.layer_dims_yaml)

        # Initialize analyzer
        analyzer = QuantizationAnalyzer(hardware_config)

        # Perform analysis
        results = analyzer.analyze_strategy(args.strategy, args.lookup_table,
                                            layer_dimensions)

        # Display results
        ResultsManager.print_results(results)

        # Save results if requested
        if args.save_results:
            ResultsManager.save_results(results, args.save_results,
                                        args.hardware_config_yaml,
                                        args.layer_dims_yaml)

        # Export LaTeX table if requested
        if args.export_latex:
            ResultsManager.save_latex_table(results, args.export_latex,
                                            args.latex_caption,
                                            args.latex_label)

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"✗ {e}")
        raise
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        raise


if __name__ == '__main__':
    main()
