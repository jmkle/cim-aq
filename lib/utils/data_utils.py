# Code for "[HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
# Kuan Wang*, Zhijian Liu*, Yujun Lin*, Ji Lin, Song Han
# {kuanwang, zhijian, yujunlin, jilin, songhan}@mit.edu

from pathlib import Path

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from lib.utils.logger import logger


def get_dataset(dataset_name,
                batch_size,
                n_worker,
                data_root='data/imagenet',
                for_inception=False,
                device=None):
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == 'cuda'

    logger.info(f'==> Preparing data for {dataset_name}..')
    if dataset_name == 'imagenet':
        traindir = Path(data_root) / 'train'
        valdir = Path(data_root) / 'val'
        assert traindir.exists(), f'{traindir} not found'
        assert valdir.exists(), f'{valdir} not found'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_size = 299 if for_inception else 224

        train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(
            str(traindir),
            transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=n_worker,
                                                   pin_memory=pin_memory)

        val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(
            str(valdir),
            transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])),
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=n_worker,
                                                 pin_memory=pin_memory)

        n_class = 1000
    elif dataset_name == 'imagenet100':
        traindir = Path(data_root) / 'train'
        valdir = Path(data_root) / 'val'
        assert traindir.exists(), f'{traindir} not found'
        assert valdir.exists(), f'{valdir} not found'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_size = 299 if for_inception else 224

        train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(
            str(traindir),
            transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=n_worker,
                                                   pin_memory=pin_memory)

        val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(
            str(valdir),
            transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])),
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=n_worker,
                                                 pin_memory=pin_memory)

        n_class = 100
    elif dataset_name == 'imagenet10':
        traindir = Path(data_root) / 'train'
        valdir = Path(data_root) / 'val'
        assert traindir.exists(), f'{traindir} not found'
        assert valdir.exists(), f'{valdir} not found'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_size = 299 if for_inception else 224

        train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(
            str(traindir),
            transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=n_worker,
                                                   pin_memory=pin_memory)

        val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(
            str(valdir),
            transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])),
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=n_worker,
                                                 pin_memory=pin_memory)

        n_class = 10
    else:
        # Add customized data here
        raise NotImplementedError
    return train_loader, val_loader, n_class


def get_split_train_dataset(dataset_name,
                            batch_size,
                            n_worker,
                            val_size,
                            train_size=None,
                            random_seed=1,
                            data_root='data/imagenet',
                            for_inception=False,
                            shuffle=True,
                            device=None):
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == 'cuda'

    if shuffle:
        index_sampler = SubsetRandomSampler
    else:
        # use the same order
        class SubsetSequentialSampler(SubsetRandomSampler):

            def __iter__(self):
                return (self.indices[i]
                        for i in torch.arange(len(self.indices)).int())

        index_sampler = SubsetSequentialSampler

    logger.info(f'==> Preparing data for {dataset_name}..')
    if dataset_name == 'imagenet':

        traindir = Path(data_root) / 'train'
        valdir = Path(data_root) / 'val'
        assert traindir.exists(), f'{traindir} not found'
        assert valdir.exists(), f'{valdir} not found'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_size = 299 if for_inception else 224
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])

        trainset = datasets.ImageFolder(str(traindir), train_transform)
        valset = datasets.ImageFolder(str(traindir), test_transform)

        n_train = len(trainset)
        indices = list(range(n_train))
        # shuffle the indices
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        assert val_size < n_train, 'val size should less than n_train'
        train_idx, val_idx = indices[val_size:], indices[:val_size]
        if train_size:
            train_idx = train_idx[:train_size]
        logger.info(f'Data: train: {len(train_idx)}, val: {len(val_idx)}')

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)

        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=batch_size,
                                                   sampler=train_sampler,
                                                   num_workers=n_worker,
                                                   pin_memory=pin_memory)
        val_loader = torch.utils.data.DataLoader(valset,
                                                 batch_size=batch_size,
                                                 sampler=val_sampler,
                                                 num_workers=n_worker,
                                                 pin_memory=pin_memory)
        n_class = 1000
    elif dataset_name == 'imagenet100':

        traindir = Path(data_root) / 'train'
        valdir = Path(data_root) / 'val'
        assert traindir.exists(), f'{traindir} not found'
        assert valdir.exists(), f'{valdir} not found'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_size = 299 if for_inception else 224
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])

        trainset = datasets.ImageFolder(str(traindir), train_transform)
        valset = datasets.ImageFolder(str(traindir), test_transform)

        n_train = len(trainset)
        indices = list(range(n_train))
        # shuffle the indices
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        assert val_size < n_train, 'val size should less than n_train'
        train_idx, val_idx = indices[val_size:], indices[:val_size]
        if train_size:
            train_idx = train_idx[:train_size]
        logger.info(f'Data: train: {len(train_idx)}, val: {len(val_idx)}')

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)

        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=batch_size,
                                                   sampler=train_sampler,
                                                   num_workers=n_worker,
                                                   pin_memory=pin_memory)
        val_loader = torch.utils.data.DataLoader(valset,
                                                 batch_size=batch_size,
                                                 sampler=val_sampler,
                                                 num_workers=n_worker,
                                                 pin_memory=pin_memory)
        n_class = 100
    else:
        raise NotImplementedError

    return train_loader, val_loader, n_class
