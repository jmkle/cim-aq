#!/usr/bin/env python3
##############################################################################
# Copyright (C) 2025 Joel Klein                                              #
# All Rights Reserved                                                        #
#                                                                            #
# This work is licensed under the terms described in the LICENSE file        #
# found in the root directory of this source tree.                           #
##############################################################################

from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn

from lib.utils.logger import logger
from lib.utils.quantize_utils import (CommonInt8ActQuant, CommonQuantConv2d,
                                      CommonQuantLinear, save_pop)

__all__ = [
    'ResNet', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
    'custom_resnet18', 'custom_resnet34', 'custom_resnet50',
    'custom_resnet101', 'custom_resnet152', 'qresnet18', 'qresnet34',
    'qresnet50', 'qresnet101', 'qresnet152'
]


def conv3x3(in_planes: int,
            out_planes: int,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1,
            conv_layer: Callable[..., nn.Module] = nn.Conv2d,
            quantization_strategy: list[list[int]] = [],
            max_bit: int = 8) -> nn.Module:
    """3x3 convolution with padding"""
    if conv_layer == nn.Conv2d:
        return conv_layer(in_planes,
                          out_planes,
                          kernel_size=3,
                          stride=stride,
                          padding=dilation,
                          groups=groups,
                          bias=False,
                          dilation=dilation)
    else:
        weight_bit_width, input_bit_width = save_pop(quantization_strategy,
                                                     max_bit)
        return conv_layer(in_planes,
                          out_planes,
                          kernel_size=3,
                          stride=stride,
                          padding=dilation,
                          groups=groups,
                          bias=False,
                          dilation=dilation,
                          weight_bit_width=weight_bit_width,
                          input_bit_width=input_bit_width)


def conv1x1(in_planes: int,
            out_planes: int,
            stride: int = 1,
            conv_layer: Callable[..., nn.Module] = nn.Conv2d,
            quantization_strategy: list[list[int]] = [],
            max_bit: int = 8) -> nn.Module:
    """1x1 convolution"""
    if conv_layer == nn.Conv2d:
        return conv_layer(in_planes,
                          out_planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False)
    else:
        weight_bit_width, input_bit_width = save_pop(quantization_strategy,
                                                     max_bit)
        return conv_layer(in_planes,
                          out_planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False,
                          weight_bit_width=weight_bit_width,
                          input_bit_width=input_bit_width)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: nn.Module | None = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Callable[..., nn.Module] | None = None,
                 conv_layer: Callable[..., nn.Module] = nn.Conv2d,
                 quantization_strategy: list[list[int]] = [],
                 max_bit: int = 8) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes,
                             planes,
                             stride,
                             conv_layer=conv_layer,
                             quantization_strategy=quantization_strategy,
                             max_bit=max_bit)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes,
                             planes,
                             conv_layer=conv_layer,
                             quantization_strategy=quantization_strategy,
                             max_bit=max_bit)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: nn.Module | None = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Callable[..., nn.Module] | None = None,
                 conv_layer: Callable[..., nn.Module] = nn.Conv2d,
                 quantization_strategy: list[list[int]] = [],
                 max_bit: int = 8) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes,
                             width,
                             conv_layer=conv_layer,
                             quantization_strategy=quantization_strategy,
                             max_bit=max_bit)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width,
                             width,
                             stride,
                             groups,
                             dilation,
                             conv_layer=conv_layer,
                             quantization_strategy=quantization_strategy,
                             max_bit=max_bit)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width,
                             planes * self.expansion,
                             conv_layer=conv_layer,
                             quantization_strategy=quantization_strategy,
                             max_bit=max_bit)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block: type[BasicBlock | Bottleneck],
                 layers: list[int],
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: list[bool] | None = None,
                 norm_layer: Callable[..., nn.Module] | None = None,
                 conv_layer: Callable[..., nn.Module] = nn.Conv2d,
                 linear_layer: Callable[..., nn.Module] = nn.Linear,
                 quantization_strategy: list[list[int]] = [],
                 max_bit: int = 8) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.conv_layer = conv_layer
        self.quantization_strategy = quantization_strategy
        self.max_bit = max_bit

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # Use signed quantization for first layer (to handle negative inputs)
        if conv_layer == nn.Conv2d:
            self.conv1 = conv_layer(3,
                                    self.inplanes,
                                    kernel_size=7,
                                    stride=2,
                                    padding=3,
                                    bias=False)
        else:
            weight_bit_width, input_bit_width = save_pop(
                quantization_strategy, max_bit)
            self.conv1 = conv_layer(3,
                                    self.inplanes,
                                    kernel_size=7,
                                    stride=2,
                                    padding=3,
                                    bias=False,
                                    weight_bit_width=weight_bit_width,
                                    input_quant=CommonInt8ActQuant,
                                    input_bit_width=input_bit_width)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block,
            64,
            layers[0],
            quantization_strategy=quantization_strategy,
            max_bit=max_bit)
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            quantization_strategy=quantization_strategy,
            max_bit=max_bit)
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            quantization_strategy=quantization_strategy,
            max_bit=max_bit)
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            quantization_strategy=quantization_strategy,
            max_bit=max_bit)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if linear_layer == nn.Linear:
            self.fc = linear_layer(512 * block.expansion, num_classes)
        else:
            weight_bit_width, input_bit_width = save_pop(
                quantization_strategy, max_bit)
            self.fc = linear_layer(512 * block.expansion,
                                   num_classes,
                                   weight_bit_width=weight_bit_width,
                                   input_bit_width=input_bit_width)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self,
                    block: type[BasicBlock | Bottleneck],
                    planes: int,
                    blocks: int,
                    stride: int = 1,
                    dilate: bool = False,
                    quantization_strategy: list[list[int]] = [],
                    max_bit: int = 8) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes,
                        planes * block.expansion,
                        stride,
                        conv_layer=self.conv_layer,
                        quantization_strategy=quantization_strategy,
                        max_bit=max_bit),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer,
                  self.conv_layer, quantization_strategy, max_bit))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer,
                      conv_layer=self.conv_layer,
                      quantization_strategy=quantization_strategy,
                      max_bit=max_bit))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# ResNet variant classes
class ResNet18(ResNet):

    def __init__(self, **kwargs) -> None:
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], **kwargs)


class ResNet34(ResNet):

    def __init__(self, **kwargs) -> None:
        super(ResNet34, self).__init__(BasicBlock, [3, 4, 6, 3], **kwargs)


class ResNet50(ResNet):

    def __init__(self, **kwargs) -> None:
        super(ResNet50, self).__init__(Bottleneck, [3, 4, 6, 3], **kwargs)


class ResNet101(ResNet):

    def __init__(self, **kwargs) -> None:
        super(ResNet101, self).__init__(Bottleneck, [3, 4, 23, 3], **kwargs)


class ResNet152(ResNet):

    def __init__(self, **kwargs) -> None:
        super(ResNet152, self).__init__(Bottleneck, [3, 8, 36, 3], **kwargs)


def _load_pretrained(model: nn.Module,
                     path: str,
                     strict: bool = False) -> nn.Module:
    """Load pretrained weights from checkpoint file."""
    logger.info(f'==> load pretrained model from {path}..')
    assert Path(path).is_file(), 'Error: no checkpoint directory found!'
    ch = torch.load(path)

    # Handle both state_dict formats (with 'state_dict' key or direct dict)
    if 'state_dict' in ch:
        ch = ch['state_dict']

    # Remove module. prefix if present
    ch = {k.replace('module.', ''): v for k, v in ch.items()}

    model.load_state_dict(ch, strict=strict)
    return model


# Custom ResNet model functions (standard precision)
def custom_resnet18(pretrained: bool = False, **kwargs) -> ResNet18:
    """ResNet-18 model from 'Deep Residual Learning for Image Recognition'"""
    model = ResNet18(**kwargs)
    if pretrained:
        model = _load_pretrained(
            model, 'pretrained/imagenet/custom_resnet18.pth.tar')
    return model


def custom_resnet34(pretrained: bool = False, **kwargs) -> ResNet34:
    """ResNet-34 model from 'Deep Residual Learning for Image Recognition'"""
    model = ResNet34(**kwargs)
    if pretrained:
        model = _load_pretrained(
            model, 'pretrained/imagenet/custom_resnet34.pth.tar')
    return model


def custom_resnet50(pretrained: bool = False, **kwargs) -> ResNet50:
    """ResNet-50 model from 'Deep Residual Learning for Image Recognition'"""
    model = ResNet50(**kwargs)
    if pretrained:
        model = _load_pretrained(
            model, 'pretrained/imagenet/custom_resnet50.pth.tar')
    return model


def custom_resnet101(pretrained: bool = False, **kwargs) -> ResNet101:
    """ResNet-101 model from 'Deep Residual Learning for Image Recognition'"""
    model = ResNet101(**kwargs)
    if pretrained:
        model = _load_pretrained(
            model, 'pretrained/imagenet/custom_resnet101.pth.tar')
    return model


def custom_resnet152(pretrained: bool = False, **kwargs) -> ResNet152:
    """ResNet-152 model from 'Deep Residual Learning for Image Recognition'"""
    model = ResNet152(**kwargs)
    if pretrained:
        model = _load_pretrained(
            model, 'pretrained/imagenet/custom_resnet152.pth.tar')
    return model


# Quantized ResNet model functions
def qresnet18(pretrained: bool = False,
              num_classes: int = 1000,
              quantization_strategy: list[list[int]] = [],
              max_bit: int = 8,
              **kwargs) -> ResNet18:
    """Quantized ResNet-18 model"""
    model = ResNet18(conv_layer=CommonQuantConv2d,
                     linear_layer=CommonQuantLinear,
                     num_classes=num_classes,
                     quantization_strategy=quantization_strategy,
                     max_bit=max_bit,
                     **kwargs)
    if pretrained:
        model = _load_pretrained(model,
                                 'pretrained/imagenet/qresnet18.pth.tar')
    return model


def qresnet34(pretrained: bool = False,
              num_classes: int = 1000,
              quantization_strategy: list[list[int]] = [],
              max_bit: int = 8,
              **kwargs) -> ResNet34:
    """Quantized ResNet-34 model"""
    model = ResNet34(conv_layer=CommonQuantConv2d,
                     linear_layer=CommonQuantLinear,
                     num_classes=num_classes,
                     quantization_strategy=quantization_strategy,
                     max_bit=max_bit,
                     **kwargs)
    if pretrained:
        model = _load_pretrained(model,
                                 'pretrained/imagenet/qresnet34.pth.tar')
    return model


def qresnet50(pretrained: bool = False,
              num_classes: int = 1000,
              quantization_strategy: list[list[int]] = [],
              max_bit: int = 8,
              **kwargs) -> ResNet50:
    """Quantized ResNet-50 model"""
    model = ResNet50(conv_layer=CommonQuantConv2d,
                     linear_layer=CommonQuantLinear,
                     num_classes=num_classes,
                     quantization_strategy=quantization_strategy,
                     max_bit=max_bit,
                     **kwargs)
    if pretrained:
        model = _load_pretrained(model,
                                 'pretrained/imagenet/qresnet50.pth.tar')
    return model


def qresnet101(pretrained: bool = False,
               num_classes: int = 1000,
               quantization_strategy: list[list[int]] = [],
               max_bit: int = 8,
               **kwargs) -> ResNet101:
    """Quantized ResNet-101 model"""
    model = ResNet101(conv_layer=CommonQuantConv2d,
                      linear_layer=CommonQuantLinear,
                      num_classes=num_classes,
                      quantization_strategy=quantization_strategy,
                      max_bit=max_bit,
                      **kwargs)
    if pretrained:
        model = _load_pretrained(model,
                                 'pretrained/imagenet/qresnet101.pth.tar')
    return model


def qresnet152(pretrained: bool = False,
               num_classes: int = 1000,
               quantization_strategy: list[list[int]] = [],
               max_bit: int = 8,
               **kwargs) -> ResNet152:
    """Quantized ResNet-152 model"""
    model = ResNet152(conv_layer=CommonQuantConv2d,
                      linear_layer=CommonQuantLinear,
                      num_classes=num_classes,
                      quantization_strategy=quantization_strategy,
                      max_bit=max_bit,
                      **kwargs)
    if pretrained:
        model = _load_pretrained(model,
                                 'pretrained/imagenet/qresnet152.pth.tar')
    return model
