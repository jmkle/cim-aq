#!/bin/bash
##############################################################################
# Copyright (C) 2025 Joel Klein                                              #
# All Rights Reserved                                                        #
#                                                                            #
# This work is licensed under the terms described in the LICENSE file        #
# found in the root directory of this source tree.                           #
##############################################################################

# Generate minimal test dataset for CIM-AQ workflow testing
# This script creates a minimal ImageNet-like dataset structure for testing

set -e

echo "Creating test dataset with ImageNet-like structure..."

# Create directory structure
mkdir -p data/test/train
mkdir -p data/test/val

# Use only first 2 classes from imagenet100.txt for fast testing
head -2 lib/utils/imagenet100.txt > /tmp/test_classes.txt

echo "Creating dataset with 2 classes for fast testing..."

# Create dataset with 2 classes, 1 image each for train and val (224x224 like ImageNet)
while IFS= read -r class_name; do
  echo "Creating test data for class: $class_name"

  # Create class directories
  mkdir -p "data/test/train/$class_name"
  mkdir -p "data/test/val/$class_name"

  # Create small dummy images (224x224 RGB images - models expect this size) using Python
  python3 -c "
import os
from PIL import Image
import numpy as np

class_name = '$class_name'

# Create RGB images (224x224) as models expect this size
for split in ['train', 'val']:
    for i in range(1):  # 1 image per class per split for ultra-fast testing
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(f'data/test/{split}/{class_name}/image_{i:03d}.JPEG')
  "
done < /tmp/test_classes.txt

echo "✅ Test dataset created successfully!"
echo "Dataset structure:"
echo "  - 2 classes for minimal binary classification testing"
echo "  - 1 image per class for train/val splits"
echo "  - Image size: 224x224 (standard size for models)"
echo "  - Total: 4 images (2 train, 2 val)"

# Clean up
rm /tmp/test_classes.txt
