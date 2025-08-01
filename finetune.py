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

import argparse
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import wandb
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import models as customized_models
from lib.utils.data_utils import get_dataset
from lib.utils.export_utils import export_models
from lib.utils.logger import logger as main_logger
from lib.utils.utils import AverageMeter, MetricsLogger, accuracy

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
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='data/imagenet', type=str)
parser.add_argument('--data_name', default='imagenet', type=str)
parser.add_argument('-j',
                    '--workers',
                    default=16,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--warmup_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual warmup epoch number (useful on restarts)')
parser.add_argument('--train_batch',
                    default=256,
                    type=int,
                    metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test_batch',
                    default=512,
                    type=int,
                    metavar='N',
                    help='test batchsize (default: 512)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate')
parser.add_argument('--lr_type',
                    default='cos',
                    type=str,
                    help='lr scheduler (exp/cos/step3/fixed)')
parser.add_argument('--schedule',
                    type=int,
                    nargs='+',
                    default=[31, 61, 91],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma',
                    type=float,
                    default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-5,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-5)')
# Checkpoints
parser.add_argument('-c',
                    '--checkpoint',
                    default='checkpoint',
                    type=str,
                    metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained',
                    action='store_true',
                    help='use pretrained model')
# Quantization
parser.add_argument('--strategy_file',
                    required=True,
                    type=str,
                    help='path to strategy file')
# Architecture
parser.add_argument('--arch',
                    '-a',
                    metavar='ARCH',
                    default='resnet50',
                    choices=model_names,
                    help='model architecture:' + ' | '.join(model_names) +
                    ' (default: resnet50)')
# Miscs
parser.add_argument('--amp', action='store_true', help='use amp')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
# Device options
parser.add_argument('--gpu_id',
                    default='1',
                    type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# W&B options
parser.add_argument('--wandb_enable',
                    action='store_true',
                    help='enable Weights & Biases logging')
parser.add_argument('--wandb_project',
                    default='haq-quantization',
                    type=str,
                    help='W&B project name')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
lr_current = state['lr']

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


def load_my_state_dict(model, state_dict):
    model_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in model_state:
            continue
        param_data = param.data
        if model_state[name].shape == param_data.shape:
            model_state[name].copy_(param_data)


def train(train_loader, model, criterion, optimizer, epoch, device):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # Get scaler for AMP
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
    for inputs, targets in pbar:
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)

        # compute output
        optimizer.zero_grad()
        if args.amp:
            with autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Scales loss and calls backward() to create scaled gradients
            scaler.scale(loss).backward()

            # Unscales gradients and calls optimizer.step()
            scaler.step(optimizer)

            # Updates the scale for next iteration
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Top1': f'{top1.avg:.4f}',
            'Top5': f'{top5.avg:.4f}',
            'Data': f'{data_time.val:.4f}s',
            'Batch': f'{batch_time.val:.4f}s'
        })

    return losses.avg, top1.avg, top5.avg


def test(val_loader, model, criterion, epoch, device):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        # switch to evaluate mode
        model.eval()

        end = time.time()
        pbar = tqdm(val_loader, desc=f'Testing Epoch {epoch+1}')
        for inputs, targets in pbar:
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.to(device), targets.to(device)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Top1': f'{top1.avg:.4f}',
                'Top5': f'{top5.avg:.4f}',
                'Data': f'{data_time.avg:.4f}s',
                'Batch': f'{batch_time.avg:.4f}s'
            })

    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state,
                    is_best,
                    checkpoint='checkpoint',
                    filename='checkpoint.pth.tar'):
    filepath = Path(checkpoint) / filename
    torch.save(state, str(filepath))
    if is_best:
        shutil.copyfile(str(filepath),
                        str(Path(checkpoint) / 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global lr_current
    global best_acc
    if epoch < args.warmup_epoch:
        lr_current = state['lr'] * args.gamma
    elif args.lr_type == 'cos':
        # cos
        lr_current = 0.5 * args.lr * (1 +
                                      math.cos(math.pi * epoch / args.epochs))
    elif args.lr_type == 'exp':
        step = 1
        decay = args.gamma
        lr_current = args.lr * (decay**(epoch // step))
    elif epoch in args.schedule:
        lr_current *= args.gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_current


if __name__ == '__main__':
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if args.resume:
        args.checkpoint = str(Path(args.resume).parent)

    if not Path(args.checkpoint).is_dir():
        Path(args.checkpoint).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, n_class = get_dataset(
        dataset_name=args.data_name,
        batch_size=args.train_batch,
        n_worker=args.workers,
        data_root=args.data)

    quantization_strategy = np.load(args.strategy_file).tolist()

    main_logger.info(f'Quantization strategy: {quantization_strategy}')

    model = models.__dict__[args.arch](
        pretrained=args.pretrained,
        num_classes=n_class,
        quantization_strategy=quantization_strategy)
    model = model.to(device)
    main_logger.info(
        f"=> created model '{args.arch}' pretrained is {args.pretrained}")
    main_logger.info(
        f'    Total params: {sum(p.numel() for p in model.parameters()) / 1000000.0:.4f}M'
    )
    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    if torch.cuda.device_count() > 1:
        if (args.arch.startswith('alexnet') or args.arch.startswith('vgg')
                or args.arch.startswith('qalexnet')
                or args.arch.startswith('qvgg')):
            model.features = torch.nn.DataParallel(model.features)
            model.to(device)
        else:
            model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    # Resume
    title = 'ImageNet-' + args.arch
    if args.resume:
        # Load checkpoint.
        main_logger.info('==> Resuming from checkpoint..')
        assert Path(
            args.resume).is_file(), 'Error: no checkpoint directory found!'
        args.checkpoint = str(Path(args.resume).parent)
        checkpoint = torch.load(args.resume, map_location=device)
        best_acc = checkpoint['best_acc']
        main_logger.info(f'Previous best accuracy: {best_acc}')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        log_path = Path(args.checkpoint) / 'log.txt'
        if log_path.is_file():
            logger = MetricsLogger(str(log_path), title=title, resume=True)
        else:
            logger = MetricsLogger(str(log_path), title=title)
            logger.set_names([
                'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.',
                'Valid Acc.', 'Train Acc5', 'Valid Acc5'
            ])
    else:
        log_path = Path(args.checkpoint) / 'log.txt'
        logger = MetricsLogger(str(log_path), title=title)
        logger.set_names([
            'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.',
            'Valid Acc.', 'Train Acc5', 'Valid Acc5'
        ])

    # Setup tensorboard writer
    tf_writer = SummaryWriter(log_dir=str(Path(args.checkpoint) / 'logs'))

    # Initialize W&B if enabled
    wandb_run = None
    if args.wandb_enable:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"finetune_{Path(args.checkpoint).name}",
            config=vars(args),
            tags=['finetuning', args.arch, args.data_name, 'quantized'])

    if args.evaluate:
        main_logger.info('\nEvaluation only')
        test_loss, test_acc, test_acc5 = test(val_loader, model, criterion,
                                              start_epoch, device)
        main_logger.info(
            f' Test Loss:  {test_loss:.8f}, Test Acc:  {test_acc:.4f}, Test Acc5: {test_acc5:.4f}'
        )
        exit()

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        main_logger.info(
            f'\nEpoch: [{epoch + 1} | {args.epochs}] LR: {lr_current:f}')

        train_loss, train_acc, train_acc5 = train(train_loader, model,
                                                  criterion, optimizer, epoch,
                                                  device)
        test_loss, test_acc, test_acc5 = test(val_loader, model, criterion,
                                              epoch, device)

        # append logger file
        logger.append([
            lr_current, train_loss, test_loss, train_acc, test_acc, train_acc5,
            test_acc5
        ])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            },
            is_best,
            checkpoint=args.checkpoint)

        # ============ TensorBoard logging ============#
        # Log the scalar values
        info = {
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'train_accuracy_top5': train_acc5,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_accuracy_top5': test_acc5,
            'learning_rate': lr_current
        }

        for tag, value in info.items():
            tf_writer.add_scalar(tag, value, epoch)

        # ============ W&B logging ============#
        if args.wandb_enable and wandb_run is not None:
            wandb_run.log(info, step=epoch)

    logger.close()

    inputs, _ = next(iter(train_loader))
    export_models(model, args.checkpoint, inputs, wandb_run)

    main_logger.info(f'Best accuracy: {best_acc}')
