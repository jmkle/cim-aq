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

import matplotlib.pyplot as plt
import numpy as np
import torch


class AverageMeter(object):

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def accumulate(self, val, n=1):
        self.sum += val
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


class MetricsLogger(object):

    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()


def accuracy(output, target, topk=(1, )):
    # Handle case where we have fewer classes than requested topk
    num_classes = output.size(1)
    topk = tuple(min(k, num_classes) for k in topk)

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_numpy(var):
    # return var.cpu().data.numpy()
    return var.cpu().detach().numpy() if var.is_cuda else var.detach().numpy()


def to_tensor(ndarray, requires_grad=False, dtype=torch.float32):
    return torch.tensor(ndarray,
                        requires_grad=requires_grad,
                        device=device,
                        dtype=dtype)


def sample_from_truncated_normal_distribution(lower, upper, mu, sigma, size=1):
    from scipy import stats
    return stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma,
                               loc=mu,
                               scale=sigma,
                               size=size)


def get_num_gen(gen):
    return sum(1 for x in gen)


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    import functools
    import operator

    return sum([
        functools.reduce(operator.mul, i.size(), 1)
        for i in model.parameters()
    ])


def measure_layer(layer, x):
    global count_ops, count_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)

    # ops_conv
    if type_name in ['Conv2d', 'CommonQuantConv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] -
                     layer.kernel_size[0]) / layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] -
                     layer.kernel_size[1]) / layer.stride[1] + 1)
        layer.in_h = x.size()[2]
        layer.in_w = x.size()[3]
        layer.out_h = out_h
        layer.out_w = out_w
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        delta_params = get_layer_param(layer)
        layer.flops = delta_ops
        layer.params = delta_params

    # ops_nonlinearity
    elif type_name in ['ReLU']:
        delta_ops = x.numel() / x.size(0)
        delta_params = get_layer_param(layer)

    # ops_pooling
    elif type_name in ['AvgPool2d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) /
                    layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) /
                    layer.stride + 1)
        delta_ops = x.size()[1] * out_w * out_h * kernel_ops
        delta_params = get_layer_param(layer)

    elif type_name in ['AdaptiveAvgPool2d']:
        delta_ops = x.size()[1] * x.size()[2] * x.size()[3]
        delta_params = get_layer_param(layer)

    # ops_linear
    elif type_name in ['Linear', 'CommonQuantLinear']:
        weight_ops = layer.weight.numel() * multi_add
        if layer.bias is not None:
            bias_ops = layer.bias.numel()
        else:
            bias_ops = 0
        layer.in_h = x.size()[1]
        layer.in_w = 1
        delta_ops = weight_ops + bias_ops
        delta_params = get_layer_param(layer)
        layer.flops = delta_ops
        layer.params = delta_params

    # ops_multihead_attention
    elif type_name in ['MultiheadAttention', 'CommonQuantMultiheadAttention']:
        # MHA input: (batch_size, seq_len, embed_dim)
        _, seq_len, embed_dim = x.size()

        # Store dimensions for state embedding
        layer.in_h = seq_len
        layer.in_w = embed_dim
        layer.seq_len = seq_len

        # Estimate MHA operations
        # Q, K, V projections: seq_len * embed_dim * (3 * embed_dim)
        qkv_ops = seq_len * embed_dim * (3 * embed_dim)

        # Attention computation: num_heads * seq_len^2 * head_dim * 2 (for Q @ K^T and V @ A)
        head_dim = embed_dim // layer.num_heads
        attention_ops = layer.num_heads * seq_len**2 * head_dim * 2

        # Output projection: seq_len * embed_dim * embed_dim
        output_ops = seq_len * embed_dim * embed_dim

        delta_ops = (qkv_ops + attention_ops + output_ops) * multi_add
        delta_params = get_layer_param(layer)

        layer.flops = delta_ops
        layer.params = delta_params

    # ops_nothing
    elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout']:
        delta_params = get_layer_param(layer)

    # unknown layer type
    else:
        delta_params = get_layer_param(layer)

    count_ops += delta_ops
    count_params += delta_params

    return delta_ops, delta_params


def measure_model(model, H, W):
    global count_ops, count_params
    count_ops = 0
    count_params = 0

    # Get the device of the first model parameter
    device = next(model.parameters()).device
    data = torch.zeros(1, 3, H, W, device=device)

    def should_measure(x):
        # Only measure actual computational layers, not Brevitas internal components
        layer_type = get_layer_info(x)

        # Brevitas quantized layers are not leaf nodes but should be measured
        brevitas_layers = [
            'CommonQuantConv2d', 'CommonQuantLinear',
            'CommonQuantMultiheadAttention'
        ]

        # Traditional layers that are leaf nodes
        leaf_measurable_types = [
            'Conv2d', 'Linear', 'BatchNorm2d', 'AdaptiveAvgPool2d',
            'Dropout2d', 'DropChannel', 'Dropout', 'ReLU', 'MaxPool2d',
            'MultiheadAttention'
        ]

        # Measure Brevitas layers even if they're not leaf nodes
        if layer_type in brevitas_layers:
            return True

        # Measure traditional layers if they are leaf nodes
        return is_leaf(x) and layer_type in leaf_measurable_types

    def modify_forward(module):
        for child in module.children():

            if should_measure(child):
                child.old_forward = child.forward

                def new_forward(m):

                    def lambda_forward(*args, **kwargs):
                        # Get the first argument as the main input tensor
                        x = args[0] if args else None
                        if x is not None:
                            measure_layer(m, x)
                        type_name = get_layer_info(m)

                        # Skip the actual forward pass and instead return a zero tensor with the expected output shape
                        if x is None:
                            # If no input tensor, can't proceed with measurement
                            raise ValueError(
                                "No input tensor provided to lambda_forward")

                        # Conv2D
                        if type_name == 'CommonQuantConv2d' and hasattr(
                                m, 'out_channels'):
                            b, _, _, _ = x.shape
                            return x.new_zeros(b, m.out_channels, m.out_h,
                                               m.out_w)
                        # Linear
                        if type_name == 'CommonQuantLinear' and hasattr(
                                m, 'out_features'):
                            b = x.shape[0]
                            return x.new_zeros(b, m.out_features)
                        # MultiheadAttention
                        if type_name == 'CommonQuantMultiheadAttention':
                            # MultiheadAttention returns (attn_output, attn_output_weights)
                            # For measurement purposes, return zero tensors with proper shape
                            attn_output = x.new_zeros(x.size())
                            attn_weights = None  # We can return None for weights when need_weights=False
                            return attn_output, attn_weights
                        # For other layers, return a zero tensor with the same shape as input
                        return x.new_zeros(x.size())

                    return lambda_forward

                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(module):
        for child in module.children():
            if hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                del child.old_forward
            else:
                restore_forward(child)

    modify_forward(model)
    model.eval()
    with torch.no_grad():
        model(data)
    restore_forward(model)

    return count_ops, count_params
