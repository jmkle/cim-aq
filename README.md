<!--
Copyright (C) 2025 Joel Klein
All Rights Reserved

This work is licensed under the terms described in the LICENSE file
found in the root directory of this source tree.
This work is based on the HAQ framework which can be found
here: https://github.com/mit-han-lab/haq/
-->

# CIM-AQ: CIM-aware Automated Quantization with Mixed Precision

[![Style](https://github.com/jmkle/cim-aq/actions/workflows/style.yml/badge.svg)](https://github.com/jmkle/cim-aq/actions/workflows/style.yml)
[![Build and Test](https://github.com/jmkle/cim-aq/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/jmkle/cim-aq/actions/workflows/build-and-test.yml)

This repository contains the PyTorch implementation of CIM-AQ: CIM-aware Automated Quantization with Mixed Precision.

CIM-AQ is based on the [HAQ framework](https://github.com/mit-han-lab/haq), modifying it to support Computing-in-Memory (CIM) architectures. The HAQ framework has been modernized, and its reward function has been adapted to minimize the latency of quantized models on CIM hardware while maintaining accuracy. Furthermore, the CIM-AQ framework includes a CIM-specific latency model that estimates the latency of quantized models on CIM hardware. This model is used during the quantization search process. Additionally, the framework was updated to use layers from [Xilinx/Brevitas](https://github.com/xilinx/brevitas) for quantization instead of the custom-designed layers. This allows for a more flexible and efficient quantization process that leverages Brevitas's capabilities for quantized neural networks. Brevitas provides easier extensibility to additional layer types and quantization schemes. It also offers the significant advantage that the resulting quantized neural networks can be exported directly to ONNX format, eliminating the need for additional conversion steps.

## Main folders and scripts

- `lib/` - Core library code (env, RL, simulator, utils)
- `models/` - Model definitions (ResNet, VGG, etc.)
- `run/` - Bash scripts and configs for running workflows
- `data/` - Symlink to datasets
- `finetune.py` - Finetuning quantized models
- `pretrain.py` - Pretraining models
- `rl_quantize.py` - RL-based quantization search

## Docker

For consistent environments and easy deployment, we provide a Docker image for CIM-AQ and a simple script to run it:

```bash
# Show all available options
./run/run_docker.sh --help

# Interactive bash session
./run/run_docker.sh

# Test workflow with synthetic data
./run/run_docker.sh --test

# Custom workflow with your config
./run/run_docker.sh my_config.yaml

# Use specific image tag and custom dataset
./run/run_docker.sh --tag main --data /path/to/imagenet my_config.yaml

# Use a specific image and run tests
./run/run_docker.sh --image ghcr.io/jmkle/cim-aq:pr-123 --test

# GPU options: all (default), none, specific devices
./run/run_docker.sh --gpu all --test
./run/run_docker.sh --gpu none --test
./run/run_docker.sh --gpu 0,1 --test
```

This project is designed to run rootless, so you can also use `podman`. Make sure `podman-docker` is installed or create an alias for docker that points to podman.

**Available Images:**

- `ghcr.io/jmkle/cim-aq:latest` (main branch)
- `ghcr.io/jmkle/cim-aq:<branch-name>` (feature branches)
- `ghcr.io/jmkle/cim-aq:pr-<number>` (PRs, auto-cleaned)

**GPU Requirements:** [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) + CUDA 12.9.1 compatible drivers

## Dependencies

The current codebase is tested under the following environment:

- Python 3.11.2
- PyTorch 2.7.1 (CUDA 12)
- Brevitas 0.12.0
- ONNX 1.18.0
- ONNX Optimizer 0.3.13
- torchvision 0.22.1
- Matplotlib 3.10.5
- SciPy 1.16.1
- Pillow 11.3.0
- TensorBoard 2.20.0
- tqdm 4.67.1
- W&B 0.21.0

You can install the required dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Continuous Integration & Deployment

CIM-AQ includes automated GitHub Actions CI/CD that builds, tests, and publishes Docker images.

**Main Workflow** ([`.github/workflows/build-and-test.yml`](.github/workflows/build-and-test.yml)):

- **Builds** Docker images with all dependencies
- **Publishes** to GitHub Container Registry
- **Tests** basic functionality and complete CIM-AQ workflow
- **Cleans up** PR containers automatically when closed

**Supporting Workflows**:

- **Code Quality** ([`.github/workflows/formatting.yml`](.github/workflows/formatting.yml)): Validates code formatting
- **Container Cleanup** ([`.github/workflows/cleanup-pr-containers.yml`](.github/workflows/cleanup-pr-containers.yml)): Removes old containers

**Available Images**: `ghcr.io/jmkle/cim-aq:latest` (main), `ghcr.io/jmkle/cim-aq:<branch>` (branches), `ghcr.io/jmkle/cim-aq:pr-<number>` (PRs)

## Dataset

If you already have the ImageNet dataset for PyTorch, you can create a link to the data folder and use it:

```bash
# prepare dataset, change the path to your own
ln -s /path/to/imagenet/ data/
```

If you do not have the ImageNet yet, you can download the ImageNet dataset and move validation images to labeled subfolders. To do this, you can use the following script:
[https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

We use a subset of ImageNet in the linear quantization search phase to save the training time, to create the link of the subset, you can use the following tool:

```bash
# prepare imagenet100 dataset
python lib/utils/make_data.py
```

## CIM-aware Automated Quantization Execution

We provide a script to run the complete CIM-aware automated quantization workflow:

```bash
bash run/run_full_workflow.sh /path/to/config.yaml
```

This script will execute the following steps:

1. **Stage 1**: Finding the best mixed precision strategy for a given model on smaller dataset (e.g., ImageNet100).
   1. **FP32 Pretraining**: Pretrain the model in full precision.
   1. **INT8 Pretraining**: Pretrain the model with INT8 quantization.
   1. **RL-based Quantization Search**: Perform the quantization search using reinforcement learning.
   1. **Mixed Precision Fine-tuning**: Fine-tune the model with the best mixed precision strategy.
   1. **Evaluation**: Evaluate the final quantized model.
1. **Stage 2**: Finetuning the quantized model on the full dataset (e.g., ImageNet).
   1. **FP32 Pretraining**: Pretrain the model in full precision.
   1. **INT8 Pretraining**: Pretrain the model with INT8 quantization.
   1. **Mixed Precision Fine-tuning**: Fine-tune the model with the best mixed precision strategy.
   1. **Evaluation**: Evaluate the final quantized model.

The workflow can be configured using the config.yaml file. A template configuration file ([`run/configs/config_template.yaml`](run/configs/config_template.yaml)) and an example configuration file ([`run/configs/example_config.yaml`](run/configs/example_config.yaml)) are provided. You can create your own configuration file based on these templates.

Furthermore, the configs with which we evaluated CIM-AQ are provided in the `run/configs/` folder:

- [`run/configs/qvgg16_imagenet.yaml`](run/configs/qvgg16_imagenet.yaml): Config for VGG16 quantization search on ImageNet
- [`run/configs/qresnet18_imagenet.yaml`](run/configs/qresnet18_imagenet.yaml): Config for ResNet18 quantization search on ImageNet

The steps in the workflow can be executed individually by running the corresponding scripts in the `run/` folder. The scripts are designed to be modular, so you can run only the steps you need.

## FP32 Pretraining

The FP32 pretraining can be run with the following command:

```bash
bash run_fp32_pretraining.sh [fp32_model] [dataset] [dataset_root] [fp32_finetune_epochs] [dataset_suffix] [learning_rate] [wandb_enable] [wandb_project] [gpu_id]
```

This script will pretrain the specified model in full precision on the given dataset. The pretrained model will be saved in the `checkpoints/<model>_pretrained_<dataset_suffix>/` directory.

The FP32 pretraining downloads pretrained models from the [torchvision model zoo](https://pytorch.org/vision/stable/models.html) and tries to applies them before starting the training.

## INT8 Pretraining

The INT8 pretraining can be run with the following command:

```bash
bash run_int8_pretraining.sh [quant_model] [fp32_model] [dataset] [dataset_root] [uniform_8bit_epochs] [force_first_last_layer] [dataset_suffix] [learning_rate] [wandb_enable] [wandb_project] [gpu_id]
```

It tries to find the pretrained FP32 model in `checkpoints/<fp32_model>_pretrained_<dataset_suffix>/` directory and uses it to pretrain the model with INT8 quantization. The pretrained INT8 model will be saved in the `checkpoints/<quant_model>_<dataset_suffix>/` directory.

## Reinforcement Learning Quantization Search

The RL-based quantization search is implemented in `rl_quantize.py`. It uses a reinforcement learning approach to find the best mixed precision strategy for a given model. The search process is guided by a reward function that tries to minimize the cost while maintaining accuracy.

It can be run with the following command:

```bash
bash run/run_rl_quantize.sh [quant_model] [dataset] [dataset_root] [max_accuracy_drop] [min_bit] [max_bit] [train_episodes] [search_finetune_epochs] [force_first_last_layer] [consider_cell_resolution] [output_suffix] [finetune_lr] [uniform_model_file] [wandb_enable] [wandb_project] [gpu_id]
```

Internally, it calls the `rl_quantize.py` script with the provided parameters. See `rl_quantize.py --help` for more details on the available options. After searching, the best quantization strategy is saved in the `save/<model>_<dataset>_<output_suffix>/best_policy.npy` file.

The reinforcement learning quantization search can take a long time, depending on the model and dataset. Therefore, two constraints can be applied to limit the search space:

1. `consider_cell_resolution`: If set to `True`, the search will consider the resolution of the cells for the weights in the model, which can significantly reduce the search space.
1. `force_first_last_layer`: If set to `True`, the first and last layers of the model will always be quantized to 8 bit precision, which can help maintain accuracy.

## Finetuning

After searching, you can use the `.npy` strategy file to finetune and evaluate:

```bash
bash run/run_mp_finetune.sh [quant_model] [dataset] [dataset_root] [finetune_epochs] [strategy_file] [output_suffix] [learning_rate] [uniform_model_file] [wandb_enable] [wandb_project] [gpu_id]
```

The `run_mp_finetuning.sh` script will finetune the model with the best mixed precision strategy found during the search phase. The finetuned model will be saved in the `checkpoints/<quant_model>_<output_suffix>/` directory.

Internally, similar to the FP32 and INT8 pretraining, it calls the `finetune.py` script with the provided parameters. You can see the available options by running:

```bash
python finetune.py --help
```

The script will also export the quantized model to ONNX QCDQ format, which can be used for further deployment or inference.

## Quantization Strategy Analysis

The `get_cost_from_lookup_table.py` script analyzes quantization strategies and provides detailed cost breakdowns:

```bash
# Basic analysis
python lib/simulator/get_cost_from_lookup_table.py --strategy best_policy.npy --lookup_table model_table.npy

# Full analysis with hardware calculations and results export
python lib/simulator/get_cost_from_lookup_table.py --strategy best_policy.npy --lookup_table model_table.npy \
    --hardware_config_yaml hardware.yaml --layer_dims_yaml layer_dims.yaml --save_results results_dir/
```

This tool provides latency analysis, crossbar operation counts, and MVM breakdowns for quantized models on CIM hardware.

## Logging and Monitoring

- Python logging for progress/status
- Progress bars via `tqdm`
- TensorBoard logs in `logs/` under checkpoint directory
- Optional: Weights & Biases logging with `--wandb_enable`

## Code Formatting

Automated formatting checks run on every push/PR. To manually check/fix formatting:

```bash
python utils/format.py         # Check formatting
python utils/format.py --fix   # Fix formatting issues
```

To see all available options, run:

```bash
python utils/format.py --help
```

## Requirements

See `requirements.txt` for details. Main requirements:

```shell
brevitas>=0.12.0
matplotlib>=3.10.5
onnx>=1.18.0
onnxoptimizer>=0.3.13
pillow>=11.3.0
scipy>=1.16.1
tensorboard>=2.20.0
torch>=2.7.1
torchvision>=0.22.1
tqdm>=4.67.1
wandb>=0.21.0
```
