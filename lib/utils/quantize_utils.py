import torch
from brevitas.core.restrict_val import RestrictValueType
from brevitas.nn import QuantConv2d, QuantLinear, QuantMultiheadAttention
from brevitas.nn.quant_layer import (ActQuantType, BiasQuantType,
                                     WeightQuantType)
from brevitas.quant import (Int8ActPerTensorFloatMSE,
                            Int8WeightPerTensorFloatMSE, Int32Bias,
                            Uint8ActPerTensorFloatMSE)

from lib.utils.logger import logger


def save_pop(quantization_strategy: list[list[int]],
             max_bit: int = 8) -> list[int]:
    """Pop the first quantization strategy if available, otherwise return a default value.
    
    This function safely extracts the first quantization configuration from a list
    of strategies. If the list is empty, it returns a default configuration with
    the specified bit-width for both weight and activation quantization.
    
    Args:
        quantization_strategy (list[list[int]]): A list of quantization configurations,
            where each configuration is a list of integers representing bit-widths
            for different quantization parameters (e.g., [weight_bits, activation_bits]).
        max_bit (int, optional): The default bit-width to use when no quantization
            strategy is available. Defaults to 8.
    
    Returns:
        list[int]: A list containing quantization bit-widths. If quantization_strategy
            is not empty, returns the first strategy. Otherwise, returns [max_bit, max_bit].
    
    Example:
        >>> quantization_strategy = [[4, 8], [8, 8], [2, 4]]
        >>> save_pop(quantization_strategy)
        [4, 8]
        >>> save_pop([])  # Empty list
        [8, 8]
        >>> save_pop([], max_bit=4)
        [4, 4]
    """
    if len(quantization_strategy) > 0:
        strategy = quantization_strategy.pop(0)
        return [int(x) for x in strategy]
    else:
        logger.warning("No quantization strategy available, using default.")
        return [int(max_bit), int(max_bit)]


class CommonInt8WeightPerTensorQuant(Int8WeightPerTensorFloatMSE):
    """Common per-tensor weight quantizer with bit-width set to a default value of 8."""
    scaling_min_val = 2e-16
    bit_width = 8
    restrict_scaling_type = RestrictValueType.FP


class CommonInt8WeightPerChannelQuant(CommonInt8WeightPerTensorQuant):
    """Common per-channel weight quantizer with bit-width set to a default value of 8."""
    scaling_per_output_channel = True
    restrict_scaling_type = RestrictValueType.FP


class CommonInt8ActQuant(Int8ActPerTensorFloatMSE):
    """Common signed act quantizer with bit-width set to a default value of 8."""
    scaling_min_val = 2e-16
    bit_width = 8
    restrict_scaling_type = RestrictValueType.FP


class CommonUint8ActQuant(Uint8ActPerTensorFloatMSE):
    """Common unsigned act quantizer with bit-width set to a default value of 8."""
    scaling_min_val = 2e-16
    bit_width = 8
    restrict_scaling_type = RestrictValueType.FP


class CommonInt32BiasQuant(Int32Bias):
    """Common bias quantizer with bit-width set to a default value of 32."""
    scaling_min_val = 2e-16
    bit_width = 32


class CommonQuantConv2d(QuantConv2d):
    """Common quantized convolution layer with default quantization parameters."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int] = 1,
                 padding: int | tuple[int, int] = 0,
                 dilation: int | tuple[int, int] = 1,
                 groups: int = 1,
                 padding_mode: str = 'zeros',
                 bias: bool | None = True,
                 weight_quant: WeightQuantType
                 | None = CommonInt8WeightPerTensorQuant,
                 weight_bit_width: int = 8,
                 bias_quant: BiasQuantType | None = CommonInt32BiasQuant,
                 input_quant: ActQuantType | None = CommonUint8ActQuant,
                 input_bit_width: int = 8,
                 output_quant: ActQuantType | None = None,
                 return_quant_tensor: bool = False,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 **kwargs) -> None:
        """Initialize a CommonQuantConv2d layer with default quantization parameters."""

        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         padding_mode=padding_mode,
                         bias=bias,
                         weight_quant=weight_quant,
                         weight_bit_width=weight_bit_width,
                         bias_quant=bias_quant,
                         input_quant=input_quant,
                         input_bit_width=input_bit_width,
                         output_quant=output_quant,
                         return_quant_tensor=return_quant_tensor,
                         device=device,
                         dtype=dtype,
                         **kwargs)


class CommonQuantLinear(QuantLinear):
    """Common quantized linear layer with default quantization parameters."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool | None = True,
                 weight_quant: WeightQuantType
                 | None = CommonInt8WeightPerTensorQuant,
                 weight_bit_width: int = 8,
                 bias_quant: BiasQuantType | None = CommonInt32BiasQuant,
                 input_quant: ActQuantType | None = CommonInt8ActQuant,
                 input_bit_width: int = 8,
                 output_quant: ActQuantType | None = None,
                 return_quant_tensor: bool = False,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 **kwargs) -> None:
        """Initialize a CommonQuantLinear layer with default quantization parameters."""

        super().__init__(in_features,
                         out_features,
                         bias=bias,
                         weight_quant=weight_quant,
                         weight_bit_width=weight_bit_width,
                         bias_quant=bias_quant,
                         input_quant=input_quant,
                         input_bit_width=input_bit_width,
                         output_quant=output_quant,
                         return_quant_tensor=return_quant_tensor,
                         device=device,
                         dtype=dtype,
                         **kwargs)


class CommonQuantMultiheadAttention(QuantMultiheadAttention):
    """Common quantized multi-head attention layer with default quantization parameters."""

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 bias: bool = True,
                 add_bias_kv: bool = False,
                 add_zero_attn: bool = False,
                 kdim: int | None = None,
                 vdim: int | None = None,
                 packed_in_proj: bool = True,
                 in_proj_input_quant: ActQuantType
                 | None = CommonInt8ActQuant,
                 in_proj_input_bit_width: int = 8,
                 in_proj_weight_quant: WeightQuantType
                 | None = CommonInt8WeightPerTensorQuant,
                 in_proj_weight_bit_width: int = 8,
                 in_proj_bias_quant: BiasQuantType
                 | None = CommonInt32BiasQuant,
                 softmax_input_quant: ActQuantType | None = None,
                 attn_output_weights_quant: ActQuantType
                 | None = CommonUint8ActQuant,
                 attn_output_weights_bit_width: int = 8,
                 q_scaled_quant: ActQuantType | None = CommonInt8ActQuant,
                 q_scaled_bit_width: int = 8,
                 k_transposed_quant: ActQuantType | None = CommonInt8ActQuant,
                 k_transposed_bit_width: int = 8,
                 v_quant: ActQuantType | None = CommonInt8ActQuant,
                 v_bit_width: int = 8,
                 out_proj_input_quant: ActQuantType
                 | None = CommonInt8ActQuant,
                 out_proj_input_bit_width: int = 8,
                 out_proj_weight_quant: WeightQuantType
                 | None = CommonInt8WeightPerTensorQuant,
                 out_proj_weight_bit_width: int = 8,
                 out_proj_bias_quant: BiasQuantType
                 | None = CommonInt32BiasQuant,
                 out_proj_output_quant: ActQuantType | None = None,
                 batch_first: bool = False,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 **kwargs) -> None:
        """Initialize a CommonQuantMultiheadAttention layer with default parameters."""

        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            packed_in_proj=packed_in_proj,
            in_proj_input_quant=in_proj_input_quant,
            in_proj_input_bit_width=in_proj_input_bit_width,
            in_proj_weight_quant=in_proj_weight_quant,
            in_proj_weight_bit_width=in_proj_weight_bit_width,
            in_proj_bias_quant=in_proj_bias_quant,
            softmax_input_quant=softmax_input_quant,
            attn_output_weights_quant=attn_output_weights_quant,
            attn_output_weights_bit_width=attn_output_weights_bit_width,
            q_scaled_quant=q_scaled_quant,
            q_scaled_bit_width=q_scaled_bit_width,
            k_transposed_quant=k_transposed_quant,
            k_transposed_bit_width=k_transposed_bit_width,
            v_quant=v_quant,
            v_bit_width=v_bit_width,
            out_proj_input_quant=out_proj_input_quant,
            out_proj_input_bit_width=out_proj_input_bit_width,
            out_proj_weight_quant=out_proj_weight_quant,
            out_proj_weight_bit_width=out_proj_weight_bit_width,
            out_proj_bias_quant=out_proj_bias_quant,
            out_proj_output_quant=out_proj_output_quant,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
            **kwargs)
