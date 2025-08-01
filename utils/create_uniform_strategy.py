#!/usr/bin/env python3
##############################################################################
# Copyright (C) 2025 Joel Klein                                              #
# All Rights Reserved                                                        #
#                                                                            #
# This work is licensed under the terms described in the LICENSE file        #
# found in the root directory of this source tree.                           #
##############################################################################

# Create a uniform quantization strategy file with specified bit-width

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

sys.path.append(str((Path(__file__).resolve().parent / "..").absolute()))

import models as customized_models
from lib.utils.logger import logger
from lib.utils.quantize_utils import CommonQuantConv2d, CommonQuantLinear

# Models
default_model_names = sorted(name for name in models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(models.__dict__[name]))

customized_models_names = sorted(
    name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(
            customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(
    description='Create uniform quantization strategy')
parser.add_argument('--arch',
                    '-a',
                    default='qvgg16',
                    type=str,
                    choices=model_names,
                    metavar='ARCH',
                    help='architecture to create strategy for: ' +
                    ' | '.join(model_names) + ' (default: qvgg16)')
parser.add_argument('--w_bit',
                    default=8,
                    type=int,
                    help='uniform weight bit-width')
parser.add_argument('--a_bit',
                    default=8,
                    type=int,
                    help='uniform activation bit-width')
parser.add_argument('--output',
                    default='save/uniform_strategies',
                    type=str,
                    help='output directory for strategy file')
parser.add_argument(
    '--force_first_last_layer',
    default=True,
    action=argparse.BooleanOptionalAction,
    help='force first and last layers to high precision (default: True)')

args = parser.parse_args()

# Create device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model instance to determine quantizable layers
logger.info(f"==> Creating model '{args.arch}' to analyze layer structure...")
with torch.no_grad():
    model = models.__dict__[args.arch](pretrained=False)
    model = model.to(device)
    model.eval()

# Get quantizable layers
quantizable_idx = []
for i, m in enumerate(model.modules()):
    if type(m) in [CommonQuantConv2d, CommonQuantLinear]:
        quantizable_idx.append(i)

logger.info(f"==> Found {len(quantizable_idx)} quantizable layers")

# Create uniform strategy
uniform_strategy = [[args.w_bit, args.a_bit]
                    for _ in range(len(quantizable_idx))]

# Fix first and last layer as special case (common practice to keep higher precision)
# Only apply if force_first_last_layer is enabled (default behavior)
force_first_last = getattr(args, 'force_first_last_layer', True)
if force_first_last:
    if len(uniform_strategy) > 0:
        uniform_strategy[0] = [8,
                               8]  # Input activation doesn't need quantization
    if len(uniform_strategy) > 1:
        uniform_strategy[-1] = [8, 8]  # Keep higher precision for output

logger.info(
    f"==> Generated uniform {args.w_bit}/{args.a_bit}-bit quantization strategy:"
)
logger.info(uniform_strategy)

# Create output directory if it doesn't exist
Path(args.output).mkdir(parents=True, exist_ok=True)
output_file = str(
    Path(args.output) / f"{args.arch}_w{args.w_bit}a{args.a_bit}.npy")

# Save strategy
np.save(output_file, uniform_strategy)
logger.info(f"==> Strategy saved to {output_file}")
