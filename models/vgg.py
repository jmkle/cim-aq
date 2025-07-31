import math
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn

from lib.utils.logger import logger
from lib.utils.quantize_utils import (CommonInt8ActQuant, CommonQuantConv2d,
                                      CommonQuantLinear, CommonUint8ActQuant,
                                      save_pop)

__all__ = [
    'VGG', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'custom_vgg11', 'custom_vgg13',
    'custom_vgg16', 'custom_vgg19', 'custom_vgg11_bn', 'custom_vgg13_bn',
    'custom_vgg16_bn', 'custom_vgg19_bn', 'qvgg11', 'qvgg13', 'qvgg16',
    'qvgg19', 'qvgg11_bn', 'qvgg13_bn', 'qvgg16_bn', 'qvgg19_bn'
]


def make_layers(cfg: list[int | str],
                batch_norm: bool = False,
                conv_layer: Callable[..., nn.Module] = nn.Conv2d,
                quantization_strategy: list[list[int]] = [],
                max_bit: int = 8) -> nn.Sequential:
    layers = []
    in_channels = 3
    is_first_layer = True
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if conv_layer == nn.Conv2d:
                conv2d = conv_layer(in_channels, v, kernel_size=3, padding=1)
            else:
                # Use signed quantization for first layer (to handle negative inputs), unsigned for others (non-negative inputs)
                weight_bit_width, input_bit_width = save_pop(
                    quantization_strategy, max_bit=max_bit)
                act_quant = CommonInt8ActQuant if is_first_layer else CommonUint8ActQuant
                conv2d = conv_layer(
                    in_channels,
                    v,
                    kernel_size=3,
                    padding=1,
                    weight_bit_width=weight_bit_width,
                    input_quant=act_quant,
                    input_bit_width=input_bit_width,
                )
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            is_first_layer = False
    return nn.Sequential(*layers)


cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13':
    [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    '19': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
        512, 'M', 512, 512, 512, 512, 'M'
    ]
}


class VGG(nn.Module):

    def __init__(self,
                 cfg_key: str,
                 batch_norm: bool = False,
                 num_classes: int = 1000,
                 init_weights: bool = True,
                 conv_layer: Callable[..., nn.Module] = nn.Conv2d,
                 linear_layer: Callable[..., nn.Module] = nn.Linear,
                 quantization_strategy: list[list[int]] = [],
                 max_bit: int = 8) -> None:
        super(VGG, self).__init__()
        self.features = make_layers(
            cfg[cfg_key],
            batch_norm=batch_norm,
            conv_layer=conv_layer,
            quantization_strategy=quantization_strategy,
            max_bit=max_bit)

        if linear_layer == nn.Linear:
            self.classifier = nn.Sequential(
                linear_layer(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                linear_layer(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                linear_layer(4096, num_classes),
            )
        else:
            weight_bit_width_1, input_bit_width_1 = save_pop(
                quantization_strategy, max_bit=max_bit)
            weight_bit_width_2, input_bit_width_2 = save_pop(
                quantization_strategy, max_bit=max_bit)
            weight_bit_width_3, input_bit_width_3 = save_pop(
                quantization_strategy, max_bit=max_bit)
            self.classifier = nn.Sequential(
                linear_layer(512 * 7 * 7,
                             4096,
                             weight_bit_width=weight_bit_width_1,
                             input_bit_width=input_bit_width_1),
                nn.ReLU(True),
                nn.Dropout(),
                linear_layer(4096,
                             4096,
                             weight_bit_width=weight_bit_width_2,
                             input_bit_width=input_bit_width_2),
                nn.ReLU(True),
                nn.Dropout(),
                linear_layer(4096,
                             num_classes,
                             weight_bit_width=weight_bit_width_3,
                             input_bit_width=input_bit_width_3),
            )

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
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


# VGG variant classes
class VGG11(VGG):

    def __init__(self, **kwargs) -> None:
        super(VGG11, self).__init__(cfg_key='11', **kwargs)


class VGG13(VGG):

    def __init__(self, **kwargs) -> None:
        super(VGG13, self).__init__(cfg_key='13', **kwargs)


class VGG16(VGG):

    def __init__(self, **kwargs) -> None:
        super(VGG16, self).__init__(cfg_key='16', **kwargs)


class VGG19(VGG):

    def __init__(self, **kwargs) -> None:
        super(VGG19, self).__init__(cfg_key='19', **kwargs)


def _load_pretrained(model: nn.Module,
                     path: str,
                     strict: bool = False) -> nn.Module:
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


def custom_vgg11(pretrained: bool = False, **kwargs) -> VGG11:
    model = VGG11(batch_norm=False, **kwargs)
    if pretrained:
        model = _load_pretrained(model,
                                 'pretrained/imagenet/custom_vgg11.pth.tar')
    return model


def custom_vgg13(pretrained: bool = False, **kwargs) -> VGG13:
    model = VGG13(batch_norm=False, **kwargs)
    if pretrained:
        model = _load_pretrained(model,
                                 'pretrained/imagenet/custom_vgg13.pth.tar')
    return model


def custom_vgg16(pretrained: bool = False, **kwargs) -> VGG16:
    model = VGG16(batch_norm=False, **kwargs)
    if pretrained:
        model = _load_pretrained(model,
                                 'pretrained/imagenet/custom_vgg16.pth.tar')
    return model


def custom_vgg19(pretrained: bool = False, **kwargs) -> VGG19:
    model = VGG19(batch_norm=False, **kwargs)
    if pretrained:
        model = _load_pretrained(model,
                                 'pretrained/imagenet/custom_vgg19.pth.tar')
    return model


# Custom model functions with batch normalization
def custom_vgg11_bn(pretrained: bool = False, **kwargs) -> VGG11:
    model = VGG11(batch_norm=True, **kwargs)
    if pretrained:
        model = _load_pretrained(
            model, 'pretrained/imagenet/custom_vgg11_bn.pth.tar')
    return model


def custom_vgg13_bn(pretrained: bool = False, **kwargs) -> VGG13:
    model = VGG13(batch_norm=True, **kwargs)
    if pretrained:
        model = _load_pretrained(
            model, 'pretrained/imagenet/custom_vgg13_bn.pth.tar')
    return model


def custom_vgg16_bn(pretrained: bool = False, **kwargs) -> VGG16:
    model = VGG16(batch_norm=True, **kwargs)
    if pretrained:
        model = _load_pretrained(
            model, 'pretrained/imagenet/custom_vgg16_bn.pth.tar')
    return model


def custom_vgg19_bn(pretrained: bool = False, **kwargs) -> VGG19:
    model = VGG19(batch_norm=True, **kwargs)
    if pretrained:
        model = _load_pretrained(
            model, 'pretrained/imagenet/custom_vgg19_bn.pth.tar')
    return model


def qvgg11(pretrained: bool = False,
           num_classes: int = 1000,
           quantization_strategy: list[list[int]] = [],
           max_bit: int = 8,
           **kwargs) -> VGG11:
    model = VGG11(batch_norm=False,
                  conv_layer=CommonQuantConv2d,
                  linear_layer=CommonQuantLinear,
                  num_classes=num_classes,
                  quantization_strategy=quantization_strategy,
                  max_bit=max_bit,
                  **kwargs)
    if pretrained:
        model = _load_pretrained(model, 'pretrained/imagenet/qvgg11.pth.tar')
    return model


def qvgg13(pretrained: bool = False,
           num_classes: int = 1000,
           quantization_strategy: list[list[int]] = [],
           max_bit: int = 8,
           **kwargs) -> VGG13:
    model = VGG13(batch_norm=False,
                  conv_layer=CommonQuantConv2d,
                  linear_layer=CommonQuantLinear,
                  num_classes=num_classes,
                  quantization_strategy=quantization_strategy,
                  max_bit=max_bit,
                  **kwargs)
    if pretrained:
        model = _load_pretrained(model, 'pretrained/imagenet/qvgg13.pth.tar')
    return model


def qvgg16(pretrained: bool = False,
           num_classes: int = 1000,
           quantization_strategy: list[list[int]] = [],
           max_bit: int = 8,
           **kwargs) -> VGG16:
    model = VGG16(batch_norm=False,
                  conv_layer=CommonQuantConv2d,
                  linear_layer=CommonQuantLinear,
                  num_classes=num_classes,
                  quantization_strategy=quantization_strategy,
                  max_bit=max_bit,
                  **kwargs)
    if pretrained:
        model = _load_pretrained(model, 'pretrained/imagenet/qvgg16.pth.tar')
    return model


def qvgg19(pretrained: bool = False,
           num_classes: int = 1000,
           quantization_strategy: list[list[int]] = [],
           max_bit: int = 8,
           **kwargs) -> VGG19:
    model = VGG19(batch_norm=False,
                  conv_layer=CommonQuantConv2d,
                  linear_layer=CommonQuantLinear,
                  num_classes=num_classes,
                  quantization_strategy=quantization_strategy,
                  max_bit=max_bit,
                  **kwargs)
    if pretrained:
        model = _load_pretrained(model, 'pretrained/imagenet/qvgg19.pth.tar')
    return model


# Quantized model functions with batch normalization
def qvgg11_bn(pretrained: bool = False,
              num_classes: int = 1000,
              quantization_strategy: list[list[int]] = [],
              max_bit: int = 8,
              **kwargs) -> VGG11:
    model = VGG11(batch_norm=True,
                  conv_layer=CommonQuantConv2d,
                  linear_layer=CommonQuantLinear,
                  num_classes=num_classes,
                  quantization_strategy=quantization_strategy,
                  max_bit=max_bit,
                  **kwargs)
    if pretrained:
        model = _load_pretrained(model,
                                 'pretrained/imagenet/qvgg11_bn.pth.tar')
    return model


def qvgg13_bn(pretrained: bool = False,
              num_classes: int = 1000,
              quantization_strategy: list[list[int]] = [],
              max_bit: int = 8,
              **kwargs) -> VGG13:
    model = VGG13(batch_norm=True,
                  conv_layer=CommonQuantConv2d,
                  linear_layer=CommonQuantLinear,
                  num_classes=num_classes,
                  quantization_strategy=quantization_strategy,
                  max_bit=max_bit,
                  **kwargs)
    if pretrained:
        model = _load_pretrained(model,
                                 'pretrained/imagenet/qvgg13_bn.pth.tar')
    return model


def qvgg16_bn(pretrained: bool = False,
              num_classes: int = 1000,
              quantization_strategy: list[list[int]] = [],
              max_bit: int = 8,
              **kwargs) -> VGG16:
    model = VGG16(batch_norm=True,
                  conv_layer=CommonQuantConv2d,
                  linear_layer=CommonQuantLinear,
                  num_classes=num_classes,
                  quantization_strategy=quantization_strategy,
                  max_bit=max_bit,
                  **kwargs)
    if pretrained:
        model = _load_pretrained(model,
                                 'pretrained/imagenet/qvgg16_bn.pth.tar')
    return model


def qvgg19_bn(pretrained: bool = False,
              num_classes: int = 1000,
              quantization_strategy: list[list[int]] = [],
              max_bit: int = 8,
              **kwargs) -> VGG19:
    model = VGG19(batch_norm=True,
                  conv_layer=CommonQuantConv2d,
                  linear_layer=CommonQuantLinear,
                  num_classes=num_classes,
                  quantization_strategy=quantization_strategy,
                  max_bit=max_bit,
                  **kwargs)
    if pretrained:
        model = _load_pretrained(model,
                                 'pretrained/imagenet/qvgg19_bn.pth.tar')
    return model
