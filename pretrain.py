# Code for "[HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
# Kuan Wang*, Zhijian Liu*, Yujun Lin*, Ji Lin, Song Han
# {kuanwang, zhijian, yujunlin, jilin, songhan}@mit.edu

import argparse
import math
import os
import random
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.models as models
from progress.bar import Bar
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

import models as customized_models
from lib.utils.data_utils import get_dataset
from lib.utils.logger import logger as main_logger
from lib.utils.quantize_utils import (QConv2d, QLinear, calibrate, dorefa,
                                      kmeans_update_model, quantize_model,
                                      set_fix_weight)
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
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)')
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
parser.add_argument('--amp', action='store_true', help='use amp')
parser.add_argument('--half_type',
                    default='O1',
                    type=str,
                    help='half type: O0/O1/O2/O3')
# Architecture
parser.add_argument('--arch',
                    '-a',
                    metavar='ARCH',
                    default='resnet50',
                    choices=model_names,
                    help='model architecture:' + ' | '.join(model_names) +
                    ' (default: resnet50)')
# Miscs
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
            # print("load%s"%name)
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

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
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
        prec1, prec5 = accuracy(outputs.detach(), targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx % 1 == 0:
            bar.suffix = \
                '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                )
            bar.next()
    bar.finish()
    return losses.avg, top1.avg


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
        bar = Bar('Processing', max=len(val_loader))
        for batch_idx, (inputs, targets) in enumerate(val_loader):
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

            # plot progress
            if batch_idx % 1 == 0:
                bar.suffix  = \
                    '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                    'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
                bar.next()
        bar.finish()
    return losses.avg, top1.avg


def save_checkpoint(state,
                    is_best,
                    checkpoint='checkpoint',
                    filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


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

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    train_loader, val_loader, n_class = get_dataset(
        dataset_name=args.data_name,
        batch_size=args.train_batch,
        n_worker=args.workers,
        data_root=args.data)

    model = models.__dict__[args.arch](pretrained=args.pretrained,
                                       num_classes=n_class)
    main_logger.info(
        f"=> creating model '{args.arch}' pretrained is {args.pretrained}")
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

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).to(device)

    # Resume
    title = 'ImageNet-' + args.arch
    if args.resume:
        # Load checkpoint.
        main_logger.info('==> Resuming from checkpoint..')
        assert os.path.isfile(
            args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume, map_location=device)
        best_acc = checkpoint['best_acc']
        main_logger.info(f'Previous best accuracy: {best_acc}')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model = model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if os.path.isfile(os.path.join(args.checkpoint, 'log.txt')):
            logger = MetricsLogger(os.path.join(args.checkpoint, 'log.txt'),
                                   title=title,
                                   resume=True)
        else:
            logger = MetricsLogger(os.path.join(args.checkpoint, 'log.txt'),
                                   title=title)
            logger.set_names([
                'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.',
                'Valid Acc.'
            ])
    else:
        logger = MetricsLogger(os.path.join(args.checkpoint, 'log.txt'),
                               title=title)
        logger.set_names([
            'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.',
            'Valid Acc.'
        ])

    # Setup tensorboard writer
    tf_writer = SummaryWriter(log_dir=os.path.join(args.checkpoint, 'logs'))
    main_logger.info(f'save the checkpoint to {args.checkpoint}')

    if args.evaluate:
        main_logger.info('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch,
                                   device)
        main_logger.info(
            f' Test Loss:  {test_loss:.8f}, Test Acc:  {test_acc:.4f}')
        exit()

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        main_logger.info(
            f'\nEpoch: [{epoch + 1} | {args.epochs}] LR: {lr_current:f}')

        train_loss, train_acc = train(train_loader, model, criterion,
                                      optimizer, epoch, device)
        test_loss, test_acc = test(val_loader, model, criterion, epoch, device)

        # append logger file
        logger.append([lr_current, train_loss, test_loss, train_acc, test_acc])

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
        # (1) Log the scalar values
        info = {
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'learning_rate': lr_current
        }

        for tag, value in info.items():
            tf_writer.add_scalar(tag, value, epoch)

    logger.close()

    main_logger.info(f'Best accuracy: {best_acc}')
