##############################################################################
# Copyright (C) 2025 Joel Klein                                              #
# All Rights Reserved                                                        #
#                                                                            #
# This work is licensed under the terms described in the LICENSE file        #
# found in the root directory of this source tree.                           #
##############################################################################

# Use CUDA base image
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set python3.11 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Copy requirements and install Python dependencies with cleanup
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge \
    && find /usr/local/lib/python3.11/dist-packages -name "*.pyc" -delete \
    && find /usr/local/lib/python3.11/dist-packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.11/dist-packages -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.11/dist-packages -name "test" -type d -exec rm -rf {} + 2>/dev/null || true \
    && rm -rf /tmp/* /var/tmp/* \
    && apt-get autoremove -y \
    && apt-get autoclean

# Set working directory
WORKDIR /workspace

# Copy the entire project
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p data pretrained/imagenet checkpoints save \
    && chmod -R 755 data pretrained checkpoints save

# Set the default command
CMD ["bash"]