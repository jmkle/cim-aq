# HAQ: Hardware-Aware Automated Quantization with Mixed Precision

## Introduction

This repo contains PyTorch implementation for paper [HAQ: Hardware-Aware Automated Quantization with Mixed Precision](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.pdf) (CVPR2019, oral)

![overview](https://hanlab.mit.edu/projects/haq/images/overview.png)

```
@inproceedings{haq,
author = {Wang, Kuan and Liu, Zhijian and Lin, Yujun and Lin, Ji and Han, Song},
title = {HAQ: Hardware-Aware Automated Quantization With Mixed Precision},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2019}
}
```

Other papers related to automated model design:

- AMC: AutoML for Model Compression and Acceleration on Mobile Devices ([ECCV 2018](https://arxiv.org/abs/1802.03494))

- ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware ([ICLR 2019](https://arxiv.org/abs/1812.00332))

## Dependencies

We evaluate this code with Pytorch 2.7.1 (cuda12) and torchvision 0.22.1. You can install dependencies with:

```bash
pip install -r requirements.txt
```

Current code base is tested under following environment:

- Python 3.11.2
- PyTorch 2.7.1
- torchvision 0.22.1
- TensorBoard 2.20.0
- tqdm 4.67.1
- W&B 0.21.0

## Dataset

If you already have the ImageNet dataset for pytorch, you could create a link to data folder and use it:

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

## Reinforcement learning search

- You can run the bash file as following to search the K-Means quantization strategy, which only quantizes the weights with K-Means to compress model size of specific model.

```bash
# K-Means quantization, for model size
bash run/run_kmeans_quantize_search.sh
```

- You can run the bash file as following to search the linear quantization strategy, which linearly quantizes both the weights and activations to reduce latency/energy of specific model.

```bash
# Linear quantization, for latency/energy
bash run/run_linear_quantize_search.sh
```

- Usage details

```bash
python rl_quantize.py --help
```

## Finetune Policy

- After searching, you can get the quantization strategy list (saved as a `.npy` file), and you can use the `--strategy_file` argument in **finetune.py** to finetune and evaluate the performance on ImageNet dataset.
- Example usage:

```bash
python finetune.py --strategy_file checkpoints/mobilenetv2/best_policy.npy ...
```

- We set the default K-Means quantization strategy searched under preserve ratio = 0.1 like:

```bash
# preserve ratio 10%
strategy = [6, 6, 5, 5, 5, 5, 4, 5, 5, 4, 5, 5, 5, 5, 5, 5, 3, 5, 4, 3, 5, 4, 3, 4, 4, 4, 2, 5, 4, 3, 3, 5, 3, 2, 5, 3, 2, 4, 3, 2, 5, 3, 2, 5, 3, 4, 2, 5, 2, 3, 4, 2, 3, 4]
```

You can follow the following bash file to finetune the K-Means quantized model to get a better performance:

```bash
bash run/run_kmeans_quantize_finetune.sh
```

- We set the default linear quantization strategy searched under preserve ratio = 0.6 like:

```bash
# preserve ratio 60%
strategy = [[8, -1], [7, 7], [5, 6], [4, 6], [5, 6], [5, 7], [5, 6], [7, 4], [4, 6], [4, 6], [7, 7], [5, 6], [4, 6], [7, 3], [5, 7], [4, 7], [7, 3], [5, 7], [4, 7], [7, 7], [4, 7], [4, 7], [6, 4], [6, 7], [4, 7], [7, 4], [6, 7], [5, 7], [7, 4], [6, 7], [5, 7], [7, 4], [6, 7], [6, 7], [6, 4], [5, 7], [6, 7], [6, 4], [5, 7], [6, 7], [7, 7], [4, 7], [7, 7], [7, 7], [4, 7], [7, 7], [7, 7], [4, 7], [7, 7], [7, 7], [4, 7], [7, 7], [8, 8]]
```

You can follow the following bash file to finetune the linear quantized model to get a better performance:

```bash
bash run/run_linear_quantize_finetune.sh
```

- Usage details

```bash
python finetune.py --help
```

## Evaluate

You can download the pretrained quantized model like this:

```bash
# download checkpoint
mkdir -p checkpoints/resnet50/
mkdir -p checkpoints/mobilenetv2/
cd checkpoints/resnet50/
wget https://hanlab.mit.edu/files/haq/resnet50_0.1_75.48.pth.tar
cd ../mobilenetv2/
wget https://hanlab.mit.edu/files/haq/qmobilenetv2_0.6_71.23.pth.tar
cd ../..
```

(If the server is down, you can download the pretrained model from google drive: [qmobilenetv2_0.6_71.23.pth.tar](https://drive.google.com/open?id=1oW1Jq17LIwcOckOzZPWDlKEhGWkZ3F_r))

You can evaluate the K-Means quantized model like this:

```bash
# evaluate K-Means quantization
bash run/run_kmeans_quantize_eval.sh
```

| Models | preserve ratio | Top1 Acc (%) | Top5 Acc (%) |
| ------------------------ | -------------- | ------------ | ------------ |
| resnet50 (original) | 1.0 | 76.15 | 92.87 |
| resnet50 (10x compress) | 0.1 | 75.48 | 92.42 |

You can evaluate the linear quantized model like this:

```bash
# evaluate linear quantization
bash run/run_linear_quantize_eval.sh
```

| Models | preserve ratio | Top1 Acc (%) | Top5 Acc (%) |
| ------------------------ | -------------- | ------------ | ------------ |
| mobilenetv2 (original) | 1.0 | 72.05 | 90.49 |
| mobilenetv2 (0.6x latency)| 0.6 | 71.23 | 90.00 |

## Logging and Monitoring

- All scripts use Python logging for progress and status.
- Progress bars are shown with `tqdm`.
- TensorBoard logs are written to `logs/` in the checkpoint directory.
- Optionally, enable Weights & Biases logging with `--wandb_enable`.

## Code Formatting

This repository uses automated code formatting to ensure consistent style. The formatting workflow automatically:

- **Checks** code formatting on every push and pull request
- **Creates PRs** with formatting fixes for the main branch
- **Commits** formatting fixes directly to feature branches

### Manual Formatting

To format code manually:

```bash
# Check formatting
cd utils && python format.py

# Fix formatting issues  
cd utils && python format.py --fix
```

## Requirements

See `requirements.txt` for up-to-date dependencies. Main requirements:

```shell
tensorboard>=2.20.0
torch>=2.7.1
torchvision>=0.22.1
tqdm>=4.67.1
wandb
```
