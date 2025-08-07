#!/usr/bin/env python3
##############################################################################
# Copyright (C) 2025 Joel Klein                                              #
# All Rights Reserved                                                        #
#                                                                            #
# This work is licensed under the terms described in the LICENSE file        #
# found in the root directory of this source tree.                           #
# This work is based on the HAQ framework which can be found                 #
# here https://github.com/mit-han-lab/haq/                                   #
##############################################################################

import math
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
from brevitas.nn.quant_layer import ActQuantType

from lib.utils.logger import logger
from lib.utils.quantize_utils import (CommonInt8ActQuant, CommonQuantConv2d,
                                      CommonQuantLinear, CommonUint8ActQuant,
                                      save_pop)

__all__ = ['MobileNetV2', 'mobilenetv2', 'qmobilenetv2']


def conv_bn(inp: int,
            oup: int,
            stride: int,
            conv_layer: Callable[..., nn.Module] = nn.Conv2d,
            input_quant: ActQuantType = CommonUint8ActQuant,
            quantization_strategy: list[list[int]] = [],
            max_bit: int = 8) -> nn.Sequential:
    if conv_layer == nn.Conv2d:
        return nn.Sequential(conv_layer(inp, oup, 3, stride, 1, bias=False),
                             nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))
    else:
        weight_bit_width, input_bit_width = save_pop(quantization_strategy,
                                                     max_bit)
        return nn.Sequential(
            conv_layer(
                inp,
                oup,
                3,
                stride,
                1,
                bias=False,
                weight_bit_width=weight_bit_width,
                input_bit_width=input_bit_width,
            ), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


def conv_1x1_bn(inp: int,
                oup: int,
                conv_layer: Callable[..., nn.Module] = nn.Conv2d,
                quantization_strategy: list[list[int]] = [],
                max_bit: int = 8) -> nn.Sequential:
    if conv_layer == nn.Conv2d:
        return nn.Sequential(conv_layer(inp, oup, 1, 1, 0, bias=False),
                             nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))
    else:
        weight_bit_width, input_bit_width = save_pop(quantization_strategy,
                                                     max_bit)
        return nn.Sequential(
            conv_layer(inp,
                       oup,
                       1,
                       1,
                       0,
                       bias=False,
                       weight_bit_width=weight_bit_width,
                       input_bit_width=input_bit_width), nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True))


def make_divisible(x: float, divisible_by: int = 8) -> int:
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):

    def __init__(self,
                 inp: int,
                 oup: int,
                 stride: int,
                 expand_ratio: int,
                 conv_layer: Callable[..., nn.Module] = nn.Conv2d,
                 quantization_strategy: list[list[int]] = [],
                 max_bit: int = 8) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            if conv_layer == nn.Conv2d:
                self.conv = nn.Sequential(
                    # dw
                    conv_layer(hidden_dim,
                               hidden_dim,
                               3,
                               stride,
                               1,
                               groups=hidden_dim,
                               bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    conv_layer(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                weight_bit_width_1, input_bit_width_1 = save_pop(
                    quantization_strategy, max_bit)
                weight_bit_width_2, input_bit_width_2 = save_pop(
                    quantization_strategy, max_bit)
                self.conv = nn.Sequential(
                    # dw
                    conv_layer(hidden_dim,
                               hidden_dim,
                               3,
                               stride,
                               1,
                               groups=hidden_dim,
                               bias=False,
                               weight_bit_width=weight_bit_width_1,
                               input_bit_width=input_bit_width_1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    conv_layer(hidden_dim,
                               oup,
                               1,
                               1,
                               0,
                               bias=False,
                               weight_bit_width=weight_bit_width_2,
                               input_bit_width=input_bit_width_2),
                    nn.BatchNorm2d(oup),
                )
        elif conv_layer == nn.Conv2d:
            self.conv = nn.Sequential(
                # pw
                conv_layer(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                conv_layer(hidden_dim,
                           hidden_dim,
                           3,
                           stride,
                           1,
                           groups=hidden_dim,
                           bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                conv_layer(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            weight_bit_width_1, input_bit_width_1 = save_pop(
                quantization_strategy, max_bit)
            weight_bit_width_2, input_bit_width_2 = save_pop(
                quantization_strategy, max_bit)
            weight_bit_width_3, input_bit_width_3 = save_pop(
                quantization_strategy, max_bit)
            self.conv = nn.Sequential(
                # pw
                conv_layer(inp,
                           hidden_dim,
                           1,
                           1,
                           0,
                           bias=False,
                           weight_bit_width=weight_bit_width_1,
                           input_bit_width=input_bit_width_1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                conv_layer(hidden_dim,
                           hidden_dim,
                           3,
                           stride,
                           1,
                           groups=hidden_dim,
                           bias=False,
                           weight_bit_width=weight_bit_width_2,
                           input_bit_width=input_bit_width_2),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                conv_layer(hidden_dim,
                           oup,
                           1,
                           1,
                           0,
                           bias=False,
                           weight_bit_width=weight_bit_width_3,
                           input_bit_width=input_bit_width_3),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):

    def __init__(self,
                 num_classes: int = 1000,
                 input_size: int = 224,
                 width_mult: float = 1.0,
                 block: Callable[..., nn.Module] = InvertedResidual,
                 conv_layer: Callable[..., nn.Module] = nn.Conv2d,
                 quantization_strategy: list[list[int]] = [],
                 max_bit: int = 8) -> None:
        super(MobileNetV2, self).__init__()
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(
            last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [
            conv_bn(3,
                    input_channel,
                    2,
                    conv_layer=conv_layer,
                    input_quant=CommonInt8ActQuant,
                    quantization_strategy=quantization_strategy,
                    max_bit=max_bit)
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            if conv_layer is not nn.Conv2d:
                output_channel = make_divisible(c * width_mult)
            else:
                output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(
                        block(input_channel,
                              output_channel,
                              s,
                              expand_ratio=t,
                              conv_layer=conv_layer,
                              quantization_strategy=quantization_strategy,
                              max_bit=max_bit))
                else:
                    self.features.append(
                        block(input_channel,
                              output_channel,
                              1,
                              expand_ratio=t,
                              conv_layer=conv_layer,
                              quantization_strategy=quantization_strategy,
                              max_bit=max_bit))
                input_channel = output_channel
        # building last several layers
        self.features.append(
            conv_1x1_bn(input_channel,
                        self.last_channel,
                        conv_layer=conv_layer,
                        quantization_strategy=quantization_strategy,
                        max_bit=max_bit))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        if conv_layer == nn.Conv2d:
            self.classifier = nn.Linear(self.last_channel, num_classes)
        else:
            weight_bit_width, input_bit_width = save_pop(
                quantization_strategy, max_bit)
            self.classifier = CommonQuantLinear(
                self.last_channel,
                num_classes,
                weight_bit_width=weight_bit_width,
                input_bit_width=input_bit_width,
            )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv2(pretrained: bool = False, **kwargs) -> MobileNetV2:
    model = MobileNetV2(**kwargs)
    if pretrained:
        # Load pretrained model.
        path = 'pretrained/imagenet/mobilenetv2-150.pth.tar'
        logger.info('==> load pretrained mobilenetv2 model..')
        assert Path(path).is_file(), 'Error: no checkpoint directory found!'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ch = torch.load(path, map_location=device)
        ch = {n.replace('module.', ''): v for n, v in ch['state_dict'].items()}
        model.load_state_dict(ch, strict=False)
    return model


def qmobilenetv2(pretrained: bool = False,
                 num_classes: int = 1000,
                 quantization_strategy: list[list[int]] = [],
                 max_bit: int = 8,
                 **kwargs) -> MobileNetV2:
    model = MobileNetV2(conv_layer=CommonQuantConv2d,
                        num_classes=num_classes,
                        quantization_strategy=quantization_strategy,
                        max_bit=max_bit,
                        **kwargs)
    if pretrained:
        # Load pretrained model.
        path = 'pretrained/imagenet/mobilenetv2-150.pth.tar'
        logger.info('==> load pretrained mobilenetv2 model..')
        assert Path(path).is_file(), 'Error: no checkpoint directory found!'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ch = torch.load(path, map_location=device)
        ch = {n.replace('module.', ''): v for n, v in ch['state_dict'].items()}
        model.load_state_dict(ch, strict=False)
    return model
