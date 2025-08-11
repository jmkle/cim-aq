#!/usr/bin/env python3
##############################################################################
# Copyright (C) 2025 Joel Klein                                              #
# All Rights Reserved                                                        #
#                                                                            #
# This work is licensed under the terms described in the LICENSE file        #
# found in the root directory of this source tree.                           #
##############################################################################

import math
from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn

from lib.utils.logger import logger
from lib.utils.quantize_utils import (CommonInt8ActQuant, CommonQuantConv2d,
                                      CommonQuantLinear,
                                      CommonQuantMultiheadAttention, save_pop)

__all__ = [
    'ViT', 'ViT_XS', 'ViT_S', 'ViT_B_16', 'ViT_B_32', 'ViT_L_16', 'ViT_L_32',
    'ViT_H_14', 'custom_vit_xs', 'custom_vit_s', 'custom_vit_b_16',
    'custom_vit_b_32', 'custom_vit_l_16', 'custom_vit_l_32', 'custom_vit_h_14',
    'qvit_xs', 'qvit_s', 'qvit_b_16', 'qvit_b_32', 'qvit_l_16', 'qvit_l_32',
    'qvit_h_14'
]

# --- ViT Configuration Dictionary ---
ViT_CONFIGS = {
    # Format: (patch_size, embed_dim, depth, num_heads, mlp_ratio)
    # patch_size: Size of image patches
    # embed_dim: Embedding dimension
    # depth: Number of transformer layers
    # num_heads: Number of attention heads
    # mlp_ratio: MLP hidden dimension ratio
    'vit_xs': (32, 192, 2, 3, 1.0),  # ViT-Extra Small (minimal for testing)
    'vit_s': (16, 256, 4, 4, 2.0),  # ViT-Small (small for testing)
    'vit_b_16': (16, 768, 12, 12, 4.0),  # ViT-Base/16
    'vit_b_32': (32, 768, 12, 12, 4.0),  # ViT-Base/32
    'vit_l_16': (16, 1024, 24, 16, 4.0),  # ViT-Large/16
    'vit_l_32': (32, 1024, 24, 16, 4.0),  # ViT-Large/32
    'vit_h_14': (14, 1280, 32, 16, 4.0),  # ViT-Huge/14
}


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU


class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        norm_layer: Callable[..., torch.nn.Module] | None = None,
        activation_layer: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        inplace: bool | None = None,
        bias: bool = True,
        dropout: float = 0.0,
        linear_layer: Callable[..., nn.Module] = nn.Linear,
        quantization_strategy: list[list[int]] = [],
        max_bit: int = 8,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            if linear_layer == nn.Linear:
                layers.append(linear_layer(in_dim, hidden_dim, bias=bias))
            else:
                weight_bit_width, input_bit_width = save_pop(
                    quantization_strategy, max_bit=max_bit)
                layers.append(
                    linear_layer(in_dim,
                                 hidden_dim,
                                 bias=bias,
                                 weight_bit_width=weight_bit_width,
                                 input_quant=CommonInt8ActQuant,
                                 input_bit_width=input_bit_width))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        if linear_layer == nn.Linear:
            layers.append(linear_layer(in_dim, hidden_channels[-1], bias=bias))
        else:
            weight_bit_width, input_bit_width = save_pop(quantization_strategy,
                                                         max_bit=max_bit)
            layers.append(
                linear_layer(in_dim,
                             hidden_channels[-1],
                             bias=bias,
                             weight_bit_width=weight_bit_width,
                             input_quant=CommonInt8ActQuant,
                             input_bit_width=input_bit_width))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self,
                 in_dim: int,
                 mlp_dim: int,
                 dropout: float,
                 linear_layer: Callable[..., nn.Module] = nn.Linear,
                 quantization_strategy: list[list[int]] = [],
                 max_bit: int = 8):
        super().__init__(in_dim, [mlp_dim, in_dim],
                         activation_layer=nn.GELU,
                         inplace=None,
                         dropout=dropout,
                         linear_layer=linear_layer,
                         quantization_strategy=quantization_strategy,
                         max_bit=max_bit)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self,
                 num_heads: int,
                 hidden_dim: int,
                 mlp_dim: int,
                 dropout: float,
                 attention_dropout: float,
                 norm_layer: Callable[...,
                                      torch.nn.Module] = partial(nn.LayerNorm,
                                                                 eps=1e-6),
                 multiheadattention_layer: Callable[
                     ..., nn.Module] = nn.MultiheadAttention,
                 linear_layer: Callable[..., nn.Module] = nn.Linear,
                 quantization_strategy: list[list[int]] = [],
                 max_bit: int = 8):
        super().__init__()
        self.num_heads = num_heads

        # Attention and MLP blocks
        self.ln_1 = norm_layer(hidden_dim)
        if multiheadattention_layer == nn.MultiheadAttention:
            self.self_attention = multiheadattention_layer(
                hidden_dim,
                num_heads,
                dropout=attention_dropout,
                batch_first=True)
        else:
            in_proj_weight_bit_width, in_proj_input_bit_width = save_pop(
                quantization_strategy, max_bit=max_bit)
            k_transposed_bit_width, q_scaled_bit_width = save_pop(
                quantization_strategy, max_bit=max_bit)
            v_bit_width, attn_output_weights_bit_width = save_pop(
                quantization_strategy, max_bit=max_bit)
            out_proj_weight_bit_width, out_proj_input_bit_width = save_pop(
                quantization_strategy, max_bit=max_bit)
            self.self_attention = multiheadattention_layer(
                hidden_dim,
                num_heads,
                dropout=attention_dropout,
                in_proj_input_bit_width=in_proj_input_bit_width,
                in_proj_weight_bit_width=in_proj_weight_bit_width,
                attn_output_weights_bit_width=attn_output_weights_bit_width,
                q_scaled_bit_width=q_scaled_bit_width,
                k_transposed_bit_width=k_transposed_bit_width,
                v_bit_width=v_bit_width,
                out_proj_input_bit_width=out_proj_input_bit_width,
                out_proj_weight_bit_width=out_proj_weight_bit_width,
                batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim,
                            mlp_dim,
                            dropout,
                            linear_layer=linear_layer,
                            quantization_strategy=quantization_strategy,
                            max_bit=max_bit)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm,
                                                             eps=1e-6),
        multiheadattention_layer: Callable[...,
                                           nn.Module] = nn.MultiheadAttention,
        linear_layer: Callable[..., nn.Module] = nn.Linear,
        quantization_strategy: list[list[int]] = [],
        max_bit: int = 8,
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length,
                        hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
                multiheadattention_layer,
                linear_layer,
                quantization_strategy=quantization_strategy,
                max_bit=max_bit,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class ViT(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        variant: str = 'vit_b_16',
        image_size: int = 224,
        num_classes: int = 1000,
        input_channels: int = 3,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        representation_size: int | None = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm,
                                                             eps=1e-6),
        conv_stem_configs: list[ConvStemConfig] | None = None,
        conv_layer: Callable[..., nn.Module] = nn.Conv2d,
        linear_layer: Callable[..., nn.Module] = nn.Linear,
        multiheadattention_layer: Callable[...,
                                           nn.Module] = nn.MultiheadAttention,
        quantization_strategy: list[list[int]] = [],
        max_bit: int = 8,
    ):
        super(ViT, self).__init__()

        if variant not in ViT_CONFIGS:
            raise ValueError(
                f"Unknown ViT variant: {variant}. Available variants: {list(ViT_CONFIGS.keys())}"
            )

        patch_size, hidden_dim, num_layers, num_heads, mlp_ratio = ViT_CONFIGS[
            variant]
        mlp_dim = int(hidden_dim * mlp_ratio)

        torch._assert(image_size % patch_size == 0,
                      "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.norm_layer = norm_layer
        self.variant = variant

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = input_channels
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                if conv_layer == nn.Conv2d:
                    seq_proj.add_module(
                        f"conv_bn_relu_{i}",
                        conv_layer(
                            in_channels=prev_channels,
                            out_channels=conv_stem_layer_config.out_channels,
                            kernel_size=conv_stem_layer_config.kernel_size,
                            stride=conv_stem_layer_config.stride,
                            bias=False,
                        ))
                else:
                    weight_bit_width, input_bit_width = save_pop(
                        quantization_strategy, max_bit=max_bit)
                    seq_proj.add_module(
                        f"conv_bn_relu_{i}",
                        conv_layer(
                            in_channels=prev_channels,
                            out_channels=conv_stem_layer_config.out_channels,
                            kernel_size=conv_stem_layer_config.kernel_size,
                            stride=conv_stem_layer_config.stride,
                            bias=False,
                            weight_bit_width=weight_bit_width,
                            input_quant=CommonInt8ActQuant,
                            input_bit_width=input_bit_width))
                seq_proj.add_module(
                    f"conv_bn_relu_{i}_bn",
                    conv_stem_layer_config.norm_layer(
                        conv_stem_layer_config.out_channels))
                seq_proj.add_module(
                    f"conv_bn_relu_{i}_relu",
                    conv_stem_layer_config.activation_layer(inplace=True))
                prev_channels = conv_stem_layer_config.out_channels
            if conv_layer == nn.Conv2d:
                seq_proj.add_module(
                    "conv_last",
                    conv_layer(in_channels=prev_channels,
                               out_channels=hidden_dim,
                               kernel_size=1))
            else:
                weight_bit_width, input_bit_width = save_pop(
                    quantization_strategy, max_bit=max_bit)
                seq_proj.add_module(
                    "conv_last",
                    conv_layer(in_channels=prev_channels,
                               out_channels=hidden_dim,
                               kernel_size=1,
                               weight_bit_width=weight_bit_width,
                               input_quant=CommonInt8ActQuant,
                               input_bit_width=input_bit_width))
            self.conv_proj: nn.Module = seq_proj
        else:
            if conv_layer == nn.Conv2d:
                self.conv_proj = conv_layer(in_channels=input_channels,
                                            out_channels=hidden_dim,
                                            kernel_size=patch_size,
                                            stride=patch_size)
            else:
                weight_bit_width, input_bit_width = save_pop(
                    quantization_strategy, max_bit=max_bit)
                self.conv_proj = conv_layer(in_channels=input_channels,
                                            out_channels=hidden_dim,
                                            kernel_size=patch_size,
                                            stride=patch_size,
                                            weight_bit_width=weight_bit_width,
                                            input_quant=CommonInt8ActQuant,
                                            input_bit_width=input_bit_width)

        seq_length = (image_size // patch_size)**2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
            multiheadattention_layer,
            linear_layer,
            quantization_strategy=quantization_strategy,
            max_bit=max_bit,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            if linear_layer == nn.Linear:
                heads_layers["head"] = linear_layer(hidden_dim, num_classes)
            else:
                weight_bit_width, input_bit_width = save_pop(
                    quantization_strategy, max_bit=max_bit)
                heads_layers["head"] = linear_layer(
                    hidden_dim,
                    num_classes,
                    weight_bit_width=weight_bit_width,
                    input_quant=CommonInt8ActQuant,
                    input_bit_width=input_bit_width)
        else:
            if linear_layer == nn.Linear:
                heads_layers["pre_logits"] = linear_layer(
                    hidden_dim, representation_size)
            else:
                weight_bit_width, input_bit_width = save_pop(
                    quantization_strategy, max_bit=max_bit)
                heads_layers["pre_logits"] = linear_layer(
                    hidden_dim,
                    representation_size,
                    weight_bit_width=weight_bit_width,
                    input_quant=CommonInt8ActQuant,
                    input_bit_width=input_bit_width)
            heads_layers["act"] = nn.Tanh()
            if linear_layer == nn.Linear:
                heads_layers["head"] = linear_layer(representation_size,
                                                    num_classes)
            else:
                weight_bit_width, input_bit_width = save_pop(
                    quantization_strategy, max_bit=max_bit)
                heads_layers["head"] = linear_layer(
                    representation_size,
                    num_classes,
                    weight_bit_width=weight_bit_width,
                    input_quant=CommonInt8ActQuant,
                    input_bit_width=input_bit_width)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[
                0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight,
                                  std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(
                self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(self.conv_proj.conv_last.weight,
                            mean=0.0,
                            std=math.sqrt(
                                2.0 / self.conv_proj.conv_last.out_channels))
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(
                self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight,
                                  std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(
            h == self.image_size,
            f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(
            w == self.image_size,
            f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x


# ViT variant classes
class ViT_XS(ViT):

    def __init__(self, **kwargs) -> None:
        super(ViT_XS, self).__init__(variant='vit_xs', **kwargs)


class ViT_S(ViT):

    def __init__(self, **kwargs) -> None:
        super(ViT_S, self).__init__(variant='vit_s', **kwargs)


class ViT_B_16(ViT):

    def __init__(self, **kwargs) -> None:
        super(ViT_B_16, self).__init__(variant='vit_b_16', **kwargs)


class ViT_B_32(ViT):

    def __init__(self, **kwargs) -> None:
        super(ViT_B_32, self).__init__(variant='vit_b_32', **kwargs)


class ViT_L_16(ViT):

    def __init__(self, **kwargs) -> None:
        super(ViT_L_16, self).__init__(variant='vit_l_16', **kwargs)


class ViT_L_32(ViT):

    def __init__(self, **kwargs) -> None:
        super(ViT_L_32, self).__init__(variant='vit_l_32', **kwargs)


class ViT_H_14(ViT):

    def __init__(self, **kwargs) -> None:
        super(ViT_H_14, self).__init__(variant='vit_h_14', **kwargs)


def _load_pretrained(model: nn.Module,
                     path: str,
                     strict: bool = False) -> nn.Module:
    logger.info(f'==> load pretrained model from {path}..')
    assert Path(path).is_file(), 'Error: no checkpoint directory found!'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ch = torch.load(path, map_location=device)

    # Handle both state_dict formats (with 'state_dict' key or direct dict)
    if 'state_dict' in ch:
        ch = ch['state_dict']

    # Remove module. prefix if present
    ch = {k.replace('module.', ''): v for k, v in ch.items()}

    model.load_state_dict(ch, strict=strict)
    return model


# Custom ViT model functions (standard precision)
def custom_vit_xs(pretrained: bool = False, **kwargs) -> ViT_XS:
    model = ViT_XS(**kwargs)
    if pretrained:
        model = _load_pretrained(model,
                                 'pretrained/imagenet/custom_vit_xs.pth.tar')
    return model


def custom_vit_s(pretrained: bool = False, **kwargs) -> ViT_S:
    model = ViT_S(**kwargs)
    if pretrained:
        model = _load_pretrained(model,
                                 'pretrained/imagenet/custom_vit_s.pth.tar')
    return model


def custom_vit_b_16(pretrained: bool = False, **kwargs) -> ViT_B_16:
    model = ViT_B_16(**kwargs)
    if pretrained:
        model = _load_pretrained(
            model, 'pretrained/imagenet/custom_vit_b_16.pth.tar')
    return model


def custom_vit_b_32(pretrained: bool = False, **kwargs) -> ViT_B_32:
    model = ViT_B_32(**kwargs)
    if pretrained:
        model = _load_pretrained(
            model, 'pretrained/imagenet/custom_vit_b_32.pth.tar')
    return model


def custom_vit_l_16(pretrained: bool = False, **kwargs) -> ViT_L_16:
    model = ViT_L_16(**kwargs)
    if pretrained:
        model = _load_pretrained(
            model, 'pretrained/imagenet/custom_vit_l_16.pth.tar')
    return model


def custom_vit_l_32(pretrained: bool = False, **kwargs) -> ViT_L_32:
    model = ViT_L_32(**kwargs)
    if pretrained:
        model = _load_pretrained(
            model, 'pretrained/imagenet/custom_vit_l_32.pth.tar')
    return model


def custom_vit_h_14(pretrained: bool = False, **kwargs) -> ViT_H_14:
    model = ViT_H_14(**kwargs)
    if pretrained:
        model = _load_pretrained(
            model, 'pretrained/imagenet/custom_vit_h_14.pth.tar')
    return model


# Quantized ViT model functions
def qvit_xs(pretrained: bool = False,
            num_classes: int = 1000,
            quantization_strategy: list[list[int]] = [],
            max_bit: int = 8,
            **kwargs) -> ViT_XS:
    model = ViT_XS(conv_layer=CommonQuantConv2d,
                   linear_layer=CommonQuantLinear,
                   multiheadattention_layer=CommonQuantMultiheadAttention,
                   num_classes=num_classes,
                   quantization_strategy=quantization_strategy,
                   max_bit=max_bit,
                   **kwargs)
    if pretrained:
        model = _load_pretrained(model, 'pretrained/imagenet/qvit_xs.pth.tar')
    return model


def qvit_s(pretrained: bool = False,
           num_classes: int = 1000,
           quantization_strategy: list[list[int]] = [],
           max_bit: int = 8,
           **kwargs) -> ViT_S:
    model = ViT_S(conv_layer=CommonQuantConv2d,
                  linear_layer=CommonQuantLinear,
                  multiheadattention_layer=CommonQuantMultiheadAttention,
                  num_classes=num_classes,
                  quantization_strategy=quantization_strategy,
                  max_bit=max_bit,
                  **kwargs)
    if pretrained:
        model = _load_pretrained(model, 'pretrained/imagenet/qvit_s.pth.tar')
    return model


def qvit_b_16(pretrained: bool = False,
              num_classes: int = 1000,
              quantization_strategy: list[list[int]] = [],
              max_bit: int = 8,
              **kwargs) -> ViT_B_16:
    model = ViT_B_16(conv_layer=CommonQuantConv2d,
                     linear_layer=CommonQuantLinear,
                     multiheadattention_layer=CommonQuantMultiheadAttention,
                     num_classes=num_classes,
                     quantization_strategy=quantization_strategy,
                     max_bit=max_bit,
                     **kwargs)
    if pretrained:
        model = _load_pretrained(model,
                                 'pretrained/imagenet/qvit_b_16.pth.tar')
    return model


def qvit_b_32(pretrained: bool = False,
              num_classes: int = 1000,
              quantization_strategy: list[list[int]] = [],
              max_bit: int = 8,
              **kwargs) -> ViT_B_32:
    model = ViT_B_32(conv_layer=CommonQuantConv2d,
                     linear_layer=CommonQuantLinear,
                     multiheadattention_layer=CommonQuantMultiheadAttention,
                     num_classes=num_classes,
                     quantization_strategy=quantization_strategy,
                     max_bit=max_bit,
                     **kwargs)
    if pretrained:
        model = _load_pretrained(model,
                                 'pretrained/imagenet/qvit_b_32.pth.tar')
    return model


def qvit_l_16(pretrained: bool = False,
              num_classes: int = 1000,
              quantization_strategy: list[list[int]] = [],
              max_bit: int = 8,
              **kwargs) -> ViT_L_16:
    model = ViT_L_16(conv_layer=CommonQuantConv2d,
                     linear_layer=CommonQuantLinear,
                     multiheadattention_layer=CommonQuantMultiheadAttention,
                     num_classes=num_classes,
                     quantization_strategy=quantization_strategy,
                     max_bit=max_bit,
                     **kwargs)
    if pretrained:
        model = _load_pretrained(model,
                                 'pretrained/imagenet/qvit_l_16.pth.tar')
    return model


def qvit_l_32(pretrained: bool = False,
              num_classes: int = 1000,
              quantization_strategy: list[list[int]] = [],
              max_bit: int = 8,
              **kwargs) -> ViT_L_32:
    model = ViT_L_32(conv_layer=CommonQuantConv2d,
                     linear_layer=CommonQuantLinear,
                     multiheadattention_layer=CommonQuantMultiheadAttention,
                     num_classes=num_classes,
                     quantization_strategy=quantization_strategy,
                     max_bit=max_bit,
                     **kwargs)
    if pretrained:
        model = _load_pretrained(model,
                                 'pretrained/imagenet/qvit_l_32.pth.tar')
    return model


def qvit_h_14(pretrained: bool = False,
              num_classes: int = 1000,
              quantization_strategy: list[list[int]] = [],
              max_bit: int = 8,
              **kwargs) -> ViT_H_14:
    model = ViT_H_14(conv_layer=CommonQuantConv2d,
                     linear_layer=CommonQuantLinear,
                     multiheadattention_layer=CommonQuantMultiheadAttention,
                     num_classes=num_classes,
                     quantization_strategy=quantization_strategy,
                     max_bit=max_bit,
                     **kwargs)
    if pretrained:
        model = _load_pretrained(model,
                                 'pretrained/imagenet/qvit_h_14.pth.tar')
    return model
