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

import torch
import torch.nn as nn
from brevitas.nn.quant_layer import ActQuantType

from lib.utils.quantize_utils import (CommonInt8ActQuant, CommonQuantConv2d,
                                      CommonQuantLinear, CommonUint8ActQuant,
                                      save_pop)

__all__ = ['MobileNet', 'mobilenet', 'qmobilenet']


def conv_bn(inp: int,
            oup: int,
            stride: int,
            conv_layer: Callable[..., nn.Module] = nn.Conv2d,
            input_quant: ActQuantType = CommonUint8ActQuant,
            quantization_strategy: list[list[int]] = [],
            max_bit: int = 8) -> nn.Sequential:
    if conv_layer == nn.Conv2d:
        return nn.Sequential(conv_layer(inp, oup, 3, stride, 1, bias=False),
                             nn.BatchNorm2d(oup), nn.ReLU(inplace=True))
    else:
        weight_bit_width, input_bit_width = save_pop(quantization_strategy,
                                                     max_bit)
        return nn.Sequential(
            conv_layer(inp,
                       oup,
                       3,
                       stride,
                       1,
                       bias=False,
                       weight_bit_width=weight_bit_width,
                       input_quant=input_quant,
                       input_bit_width=input_bit_width), nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True))


def conv_dw(inp: int,
            oup: int,
            stride: int,
            conv_layer: Callable[..., nn.Module] = nn.Conv2d,
            quantization_strategy: list[list[int]] = [],
            max_bit: int = 8) -> nn.Sequential:
    if conv_layer == nn.Conv2d:
        return nn.Sequential(
            conv_layer(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            conv_layer(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
        )
    else:
        weight_bit_width_1, input_bit_width_1 = save_pop(
            quantization_strategy, max_bit)
        weight_bit_width_2, input_bit_width_2 = save_pop(
            quantization_strategy, max_bit)
        return nn.Sequential(
            conv_layer(inp,
                       inp,
                       3,
                       stride,
                       1,
                       groups=inp,
                       bias=False,
                       weight_bit_width=weight_bit_width_1,
                       input_bit_width=input_bit_width_1),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            conv_layer(inp,
                       oup,
                       1,
                       1,
                       0,
                       bias=False,
                       weight_bit_width=weight_bit_width_2,
                       input_bit_width=input_bit_width_2),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
        )


class MobileNet(nn.Module):

    def __init__(self,
                 num_classes: int = 1000,
                 conv_layer: Callable[..., nn.Module] = nn.Conv2d,
                 profile: str = 'normal',
                 w_mul: float = 1.,
                 quantization_strategy: list[list[int]] = [],
                 max_bit: int = 8) -> None:
        super(MobileNet, self).__init__()

        # original
        if profile == 'normal':
            in_planes = 32
            cfg = [
                64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512,
                512, (1024, 2), 1024
            ]
        else:
            raise NotImplementedError

        if conv_layer is not nn.Conv2d and w_mul == 2.:
            in_planes = 64
            cfg = [
                128, (256, 2), 256, (512, 2), 512, (1024, 2), 1024, 1024, 1024,
                1024, 1024, (2048, 2), 2048
            ]

        # Use signed quantization for first layer (to handle negative inputs)
        self.conv1 = conv_bn(3,
                             in_planes,
                             stride=2,
                             conv_layer=conv_layer,
                             input_quant=CommonInt8ActQuant,
                             quantization_strategy=quantization_strategy,
                             max_bit=max_bit)

        self.features = self._make_layers(in_planes, cfg, conv_layer,
                                          quantization_strategy, max_bit)

        if conv_layer == nn.Conv2d:
            self.classifier = nn.Sequential(nn.Linear(cfg[-1], num_classes), )
        else:
            weight_bit_width, input_bit_width = save_pop(
                quantization_strategy, max_bit)
            self.classifier = nn.Sequential(
                CommonQuantLinear(cfg[-1],
                                  num_classes,
                                  weight_bit_width=weight_bit_width,
                                  input_bit_width=input_bit_width), )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.features(x)
        x = x.mean(3).mean(2)  # global average pooling

        x = self.classifier(x)
        return x

    def _make_layers(self, in_planes: int, cfg: list,
                     conv_layer: Callable[..., nn.Module],
                     quantization_strategy: list[list[int]],
                     max_bit: int) -> nn.Sequential:
        layers = []
        for x in cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(
                conv_dw(in_planes,
                        out_planes,
                        stride,
                        conv_layer=conv_layer,
                        quantization_strategy=quantization_strategy,
                        max_bit=max_bit))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or type(m) == CommonQuantConv2d:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) or type(m) == CommonQuantLinear:
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet(pretrained: bool = False, **kwargs) -> MobileNet:
    model = MobileNet(**kwargs)
    if pretrained:
        # Load pretrained model.
        raise NotImplementedError
    return model


def qmobilenet(pretrained: bool = False,
               quantization_strategy: list[list[int]] = [],
               max_bit: int = 8,
               **kwargs) -> MobileNet:
    model = MobileNet(conv_layer=CommonQuantConv2d,
                      quantization_strategy=quantization_strategy,
                      max_bit=max_bit,
                      **kwargs)
    if pretrained:
        # Load pretrained model.
        raise NotImplementedError
    return model
