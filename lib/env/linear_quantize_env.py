# Code for "[HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
# Kuan Wang*, Zhijian Liu*, Yujun Lin*, Ji Lin, Song Han
# {kuanwang, zhijian, yujunlin, jilin, songhan}@mit.edu

import math
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import yaml
from brevitas import config
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm

import models as customized_models
from lib.utils.data_utils import get_split_train_dataset
from lib.utils.logger import logger
from lib.utils.quantize_utils import (CommonQuantConv2d, CommonQuantLinear,
                                      CommonQuantMultiheadAttention)
from lib.utils.utils import AverageMeter, accuracy, measure_model

config.IGNORE_MISSING_KEYS = True  # Ignore missing keys in brevitas quantization layers

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
logger.info(f'Support models: {model_names}')


class LinearQuantizeEnv:

    def __init__(self, args):
        # default setting
        self.quantizable_layer_types = [
            CommonQuantConv2d, CommonQuantLinear, CommonQuantMultiheadAttention
        ]

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        # Set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # save options
        self.arch = args.arch
        self.cur_ind = 0
        self.quantization_strategy = []  # quantization strategy

        # Crossbar cell resolution constraint
        self.consider_cell_resolution = args.consider_cell_resolution
        self.cell_resolution = 1  # Default to 1 (no constraint)
        if self.consider_cell_resolution:
            self._load_hardware_config()

        self.finetune_lr = args.finetune_lr
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.n_data_worker = args.n_worker
        self.batch_size = args.data_bsize
        self.data_type = args.dataset
        self.data_root = args.dataset_root
        self.val_size = args.val_size
        self.train_size = args.train_size
        self.finetune_gamma = args.finetune_gamma
        self.finetune_lr = args.finetune_lr
        self.finetune_flag = args.finetune_flag
        self.finetune_epoch = args.finetune_epoch
        self.amp = args.amp

        # Initialize AMP scaler if enabled
        self.scaler = None
        if self.amp:
            self.scaler = GradScaler()

        # force first/last layer precision
        self.force_first_last_layer = args.force_first_last_layer

        # options from args
        self.min_bit = args.min_bit
        self.max_bit = args.max_bit
        self.orig_bit = args.orig_bit * 1.
        self.last_weight_action = self.max_bit
        self.last_activation_action = self.max_bit
        self.action_radio_button = True

        self.is_inception = args.arch.startswith('inception')
        self.is_imagenet = ('imagenet' in self.data_type)
        self.use_top5 = args.use_top5

        # init reward
        self.best_reward = -math.inf

        # prepare data
        self._init_data()

        # create model
        self.create_model()
        self.model_for_measure = deepcopy(self.model)
        self.pretrained_model = deepcopy(self.model.state_dict())
        total_params = sum(p.numel()
                           for p in self.model.parameters()) / 1000000.0
        logger.info(f'==> Total params: {total_params:.4f}M')
        cudnn.benchmark = True

        # build index
        self._build_index()
        self.n_quantizable_layer = len(self.quantizable_idx)

        self.model.load_state_dict(self.pretrained_model, strict=True)
        self.model = self.model.to(self.device)
        self._finetune(
            self.train_loader, self.model, epochs=1,
            verbose=False)  # Finetune for one epoch to initialize scales
        self.org_acc = self._validate(self.val_loader, self.model)
        self.target_acc = self.org_acc - args.acc_drop

        # build embedding (static part), same as pruning
        self._build_state_embedding()

        # mode
        self.cost_mode = 'crossbar'
        self.simulator_batch = 1
        self.cost_lookuptable = self._get_lookuptable()

        # restore weight
        self.reset()
        logger.info(
            f'=> original acc: {self.org_acc:.4f}% on split dataset(train: {self.train_size:7d}, val: {self.val_size:7d})'
        )
        logger.info(f'=> original cost: {self._org_cost():.4f}')

    def create_model(self):
        self.model = models.__dict__[self.arch](
            pretrained=True,
            num_classes=self.n_class,
            quantization_strategy=self.quantization_strategy.copy(),
            max_bit=self.max_bit)

        if torch.cuda.device_count() > 1:
            if self.arch.startswith('alexnet') or self.arch.startswith(
                    'vgg') or self.arch.startswith(
                        'qalexnet') or self.arch.startswith('qvgg'):
                self.model.features = torch.nn.DataParallel(
                    self.model.features)
                self.model.to(self.device)
            else:
                self.model = torch.nn.DataParallel(self.model).to(self.device)
        else:
            self.model = self.model.to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.finetune_lr,
                                   momentum=0.9,
                                   weight_decay=4e-5)

    def adjust_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.finetune_gamma

    def step(self, action):
        action = self._action_wall(action)  # convert action to valid bit width

        if self.action_radio_button:
            self.last_weight_action = action
        else:
            self.last_activation_action = action
            self.quantization_strategy.append(
                [self.last_weight_action,
                 self.last_activation_action])  # save action to strategy

        # all the actions are made
        if self._is_final_layer() and (not self.action_radio_button):
            # Force the first and last layer to be 8 bits if specified
            if self.force_first_last_layer:
                self._keep_first_last_layer()

            assert len(self.quantization_strategy) == len(
                self.quantizable_idx), 'Quantization strategy length mismatch'
            cost = self._cur_cost()
            cost_ratio = cost / self._org_cost()

            self.create_model()  # create model with new strategy
            if self.finetune_flag:
                self._finetune(self.train_loader,
                               self.model,
                               epochs=self.finetune_epoch,
                               verbose=False)
                acc = self._validate(self.val_loader, self.model)
            else:
                acc = self._validate(self.val_loader, self.model)

            reward = self.reward(acc, cost_ratio)

            info_set = {
                'cost_ratio': cost_ratio,
                'accuracy': acc,
                'cost': cost
            }

            if reward > self.best_reward:
                self.best_reward = reward
                logger.info(
                    f'New best policy: {self.quantization_strategy}, reward: {self.best_reward:.4f}, acc: {acc:.4f}, cost_ratio: {cost_ratio:.4f}'
                )
            else:
                logger.warning(
                    f'No better policy found: {self.quantization_strategy}, reward: {reward:.4f}, acc: {acc:.4f}, cost_ratio: {cost_ratio:.4f}'
                )

            obs = self.layer_embedding[
                self.cur_ind, :].copy()  # actually the same as the last state
            done = True
            self.action_radio_button = not self.action_radio_button
            return obs, reward, done, info_set

        cost = self._cur_cost()
        info_set = {'cost': cost}
        reward = 0
        done = False

        if self.action_radio_button:
            self.layer_embedding[self.cur_ind][-1] = 0.0
        else:
            self.cur_ind += 1  # the index of next layer
            self.layer_embedding[self.cur_ind][-1] = 1.0
        self.layer_embedding[self.cur_ind][-2] = float(action) / float(
            self.max_bit)
        self.layer_embedding[self.cur_ind][-1] = float(
            self.action_radio_button)
        # build next state (in-place modify)
        obs = self.layer_embedding[self.cur_ind, :].copy()
        self.action_radio_button = not self.action_radio_button
        return obs, reward, done, info_set

    # for quantization
    def reward(self, acc, cost_ratio=None):
        if self.target_acc is not None:
            # Calculate the base reward from latency reduction
            if cost_ratio is None:
                current_cost = self._cur_cost()
                original_cost = self._org_cost()
                cost_ratio = current_cost / original_cost

            if acc < self.target_acc:
                # Strong penalty for going below target accuracy
                accuracy_gap = self.target_acc - acc
                return -10.0 * accuracy_gap
            else:
                # Strong positive reward for aggressive quantization when accuracy is maintained
                # Use linear scaling instead of squared to avoid extreme values
                latency_reward = 100.0 * (1.0 / cost_ratio - 1.0)

                # Small bonus for accuracy above target
                acc_bonus = 0.1 * (acc - self.target_acc)

                return latency_reward + acc_bonus
        return (acc - self.org_acc) * 0.1

    def reset(self):
        # restore env by resetting indices
        self.cur_ind = 0
        self.quantization_strategy = []  # quantization strategy
        obs = self.layer_embedding[0].copy()
        return obs

    def _is_final_layer(self):
        return self.cur_ind == len(self.quantizable_idx) - 1

    def _keep_first_last_layer(self):
        self.quantization_strategy[0][0] = 8
        self.quantization_strategy[0][1] = 8
        self.quantization_strategy[-1][0] = 8
        self.quantization_strategy[-1][1] = 8

    def _load_hardware_config(self):
        """Load hardware configuration and extract cell resolution."""
        config_path = Path(__file__).resolve(
        ).parent.parent / 'simulator' / 'hardware_config.yaml'
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.cell_resolution = config['crossbar']['resolution_weight_bits']
            logger.info(
                f'=> Loaded crossbar cell resolution: {self.cell_resolution} bits'
            )
        except FileNotFoundError:
            logger.warning(
                f'Hardware config file not found at {config_path}. Using default resolution of 1.'
            )
            self.cell_resolution = 1
        except KeyError as e:
            logger.warning(
                f'Missing key in hardware config: {e}. Using default resolution of 1.'
            )
            self.cell_resolution = 1

    def _get_valid_bit_widths(self):
        """Get list of valid bit widths based on cell resolution constraint."""
        if not self.consider_cell_resolution:
            return list(range(int(self.min_bit), int(self.max_bit) + 1))

        valid_bits = []
        for bit in range(int(self.min_bit), int(self.max_bit) + 1):
            if bit % self.cell_resolution == 0:
                valid_bits.append(bit)

        # Ensure that we have at least one option
        if not valid_bits:
            # This can only happen if the cell resolution is bigger than the max_bit
            logger.warning(
                f'Cell resolution ({self.cell_resolution}) is larger than max_bit ({self.max_bit}). Using max_bit as only valid option.'
            )
            valid_bits.append(int(self.max_bit))

        return sorted(valid_bits)

    def _action_wall(self, action):
        assert len(self.quantization_strategy
                   ) == self.cur_ind, 'Quantization strategy length mismatch'
        # limit the action to certain range
        action = float(action)

        if self.consider_cell_resolution and self.action_radio_button:
            # Only apply cell resolution constraint to weight quantization (when action_radio_button is True)
            # Map continuous action to valid discrete bit widths for weights
            valid_bits = self._get_valid_bit_widths()

            # Create equal stride length for each valid bit width using index mapping
            num_valid_bits = len(valid_bits)
            action_mapped = int(num_valid_bits * action)
            action_idx = min(action_mapped, num_valid_bits - 1)
            quantized_action = valid_bits[action_idx]

            return quantized_action
        else:
            # Original logic for activation quantization or when cell resolution is not considered
            min_bit, max_bit = self.bound_list[self.cur_ind]
            lbound, rbound = min_bit - 0.5, max_bit + 0.5  # same stride length for each bit
            action = (rbound - lbound) * action + lbound
            action = int(np.round(action, 0))
            return action

    def _cur_cost(self):
        cur_cost = 0.
        # quantized
        for i, n_bit in enumerate(self.quantization_strategy):
            cur_cost += self.cost_lookuptable[i, n_bit[0] - 1, n_bit[1] - 1]
        return cur_cost

    def _org_cost(self):
        org_cost = 0
        for i in range(self.cost_lookuptable.shape[0]):
            org_cost += self.cost_lookuptable[i,
                                              int(self.orig_bit - 1),
                                              int(self.orig_bit - 1)]
        return org_cost

    def _init_data(self):
        self.train_loader, self.val_loader, self.n_class = get_split_train_dataset(
            self.data_type,
            self.batch_size,
            self.n_data_worker,
            data_root=self.data_root,
            val_size=self.val_size,
            train_size=self.train_size,
            for_inception=self.is_inception)

    def _build_index(self):
        self.quantizable_idx = []
        self.bound_list = []
        self.layer_component_map = {
        }  # Maps quantizable_idx to (module_idx, component_idx or None)

        quantizable_layer_count = 0

        for module_idx, m in enumerate(self.model.modules()):
            if type(m) in self.quantizable_layer_types:
                if type(m) == CommonQuantMultiheadAttention:
                    # MHA: Add 4 entries for internal components
                    for component_idx in range(4):
                        self.quantizable_idx.append(quantizable_layer_count)
                        self.layer_component_map[quantizable_layer_count] = (
                            module_idx, component_idx)
                        self.bound_list.append((self.min_bit, self.max_bit))
                        quantizable_layer_count += 1
                else:
                    self.quantizable_idx.append(quantizable_layer_count)
                    self.layer_component_map[quantizable_layer_count] = (
                        module_idx, None)
                    self.bound_list.append((self.min_bit, self.max_bit))
                    quantizable_layer_count += 1

        logger.info(f'=> Final bound list: {self.bound_list}')
        logger.info(
            f'=> Total quantizable components: {len(self.quantizable_idx)}')

    def _build_regular_layer_state(self, m, ind):
        """Build state for regular layers (Conv2d or Linear)."""
        this_state = []
        if type(m) in [nn.Conv2d, CommonQuantConv2d]:
            this_state.append([
                int(m.in_channels == m.groups)
            ])  # layer type, 0 for normal conv, 1 for conv_dw
            this_state.append([m.in_channels])  # in channels
            this_state.append([m.out_channels])  # out channels
            this_state.append([m.stride[0]])  # stride
            this_state.append([m.kernel_size[0]])  # kernel size
            this_state.append([np.prod(m.weight.size())])  # weight size
            this_state.append([m.in_w * m.in_h])  # input feature_map_size
        elif type(m) in [nn.Linear, CommonQuantLinear]:
            this_state.append([2.])  # layer type, 2 for fc
            this_state.append([m.in_features])  # in channels
            this_state.append([m.out_features])  # out channels
            this_state.append([0.])  # stride
            this_state.append([1.])  # kernel size
            this_state.append([np.prod(m.weight.size())])  # weight size
            this_state.append([m.in_w * m.in_h])  # input feature_map_size

        this_state.append([ind])  # index
        this_state.append([1.])  # bits, 1 is the max bit
        this_state.append([1.])  # action radio button, 1 is the weight action

        return this_state

    def _build_mha_component_state(self, m, component_idx, ind):
        """Build state for MHA components."""
        this_state = []
        embed_dim = m.embed_dim
        num_heads = m.num_heads
        head_dim = embed_dim // num_heads
        seq_len = getattr(m, 'seq_len', 50)  # From measurement or default

        if component_idx == 0:
            this_state.append([
                2.
            ])  # layer type, 2 for MHA QKV projection (as it is a FC layer)
            this_state.append([embed_dim])  # in features (embed_dim)
            this_state.append([embed_dim * 3
                               ])  # out features (Q, K, V combined)
            this_state.append([0.])  # stride (not applicable for FC)
            this_state.append([1.])  # kernel size (not applicable for FC)
            this_state.append([embed_dim * 3 * embed_dim
                               ])  # weight size (Q, K, V combined)
            this_state.append([[
                seq_len * embed_dim
            ]])  # input feature_map_size (seq_len * embed_dim)
        elif component_idx == 1:
            this_state.append(
                [3.]
            )  # layer type, 3 for MHA attention computation (Q @ K^T) (MatMul)
            this_state.append([head_dim])  # feature dimension (head_dim)
            this_state.append([seq_len])  # output dimension (seq_len)
            this_state.append([0.])  # stride (not applicable for MatMul)
            this_state.append([1.])  # kernel size (not applicable for MatMul)
            this_state.append([num_heads * seq_len * head_dim
                               ])  # Q tensor size (first operand)
            this_state.append([[num_heads * head_dim * seq_len]
                               ])  # K^T tensor size (second operand)
        elif component_idx == 2:
            this_state.append([
                3.
            ])  # layer type, 3 for MHA attention output (V @ O) (MatMul)
            this_state.append([seq_len])  # feature dimension (seq_len)
            this_state.append([head_dim])  # output dimension (head_dim)
            this_state.append([0.])  # stride (not applicable for MatMul)
            this_state.append([1.])  # kernel size (not applicable for MatMul)
            this_state.append([num_heads * seq_len * seq_len
                               ])  # attention output size (first operand)
            this_state.append([[num_heads * seq_len * head_dim]
                               ])  # V tensor size (second operand)
        elif component_idx == 3:
            this_state.append([
                2.
            ])  # layer type, 2 for MHA output projection (as it is a FC layer)
            this_state.append([embed_dim])  # in channels (embed_dim)
            this_state.append([embed_dim])  # out channels (embed_dim)
            this_state.append([0.])  # stride (not applicable for FC)
            this_state.append([1.])  # kernel size (not applicable for FC)
            this_state.append([embed_dim * embed_dim])  # weight size (output)
            this_state.append([[
                seq_len * embed_dim
            ]])  # input feature_map_size (seq_len * embed_dim)
        else:
            raise ValueError(
                f'Invalid component_idx {component_idx} for MHA state building'
            )

        this_state.append([ind])  # index
        this_state.append([1.])  # bits, 1 is the max bit
        this_state.append([1.])  # action radio button, 1 is the weight action
        return this_state

    def _normalize_embeddings(self, layer_embedding):
        """Embedding normalization"""
        logger.info(
            f'=> Embedding shape (n_components * n_dim): {layer_embedding.shape}'
        )
        assert len(layer_embedding.shape) == 2, layer_embedding.shape

        # Vectorized normalization
        fmin = layer_embedding.min(axis=0)
        fmax = layer_embedding.max(axis=0)
        mask = (fmax - fmin) > 0
        layer_embedding[:, mask] = (layer_embedding[:, mask] -
                                    fmin[mask]) / (fmax[mask] - fmin[mask])

        return layer_embedding

    def _build_state_embedding(self):
        # measure model for input
        if self.is_imagenet:
            measure_model(self.model_for_measure, 224, 224)
        else:
            measure_model(self.model_for_measure, 32, 32)
        # build the static part of the state embedding
        layer_embedding = []
        module_list = list(self.model_for_measure.modules())

        for ind in self.quantizable_idx:
            module_idx, component_idx = self.layer_component_map[ind]
            m = module_list[module_idx]

            if component_idx is None:
                # Regular layer (Conv2d or Linear)
                this_state = self._build_regular_layer_state(m, ind)
            else:
                # MHA component
                this_state = self._build_mha_component_state(
                    m, component_idx, ind)

            layer_embedding.append(np.hstack(this_state))

        # Normalize embeddings
        self.layer_embedding = self._normalize_embeddings(
            np.array(layer_embedding, 'float'))

    def _get_lookuptable(self):

        lookup_table_folder = 'lib/simulator/lookup_tables/'
        Path(lookup_table_folder).mkdir(parents=True, exist_ok=True)
        if self.cost_mode == 'crossbar':
            fname = lookup_table_folder + self.arch + '_batch_' + str(
                self.simulator_batch) + '_latency_table.npy'
        else:
            # add your own cost lookuptable here
            raise NotImplementedError

        if Path(fname).is_file():
            logger.info(f'load latency table : {fname}')
            latency_list = np.load(fname)
            logger.debug(f'Latency table contents: {latency_list}')
        else:
            # you can put your own simulator/lookuptable here
            raise NotImplementedError
        return latency_list.copy()

    def _finetune(self, train_loader, model, epochs=1, verbose=True):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        best_acc = 0.

        # switch to train mode
        model.train()
        end = time.time()
        t1 = time.time()

        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{epochs}')
            for inputs, targets in pbar:
                input_var, target_var = inputs.to(self.device), targets.to(
                    self.device)

                # measure data loading time
                data_time.update(time.time() - end)

                self.optimizer.zero_grad()

                # compute output
                with autocast(device_type=self.device.type, enabled=self.amp):
                    output = model(input_var)
                    loss = self.criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.detach(),
                                        target_var,
                                        topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                # compute gradient
                if self.amp and self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    # do SGD step
                    self.optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # update progress bar
                pbar.set_postfix({
                    'Loss': f'{losses.avg:.4f}',
                    'Top1': f'{top1.avg:.4f}',
                    'Top5': f'{top5.avg:.4f}',
                    'Data': f'{data_time.val:.4f}s',
                    'Batch': f'{batch_time.val:.4f}s'
                })

            if self.use_top5:
                if top5.avg > best_acc:
                    best_acc = top5.avg
            else:
                if top1.avg > best_acc:
                    best_acc = top1.avg
            self.adjust_learning_rate()
        t2 = time.time()
        if verbose:
            logger.info(
                f'* Test loss: {losses.avg:.4f}  top1: {top1.avg:.4f}  top5: {top5.avg:.4f}  time: {t2 - t1:.4f}'
            )
        return best_acc

    def _validate(self, val_loader, model, verbose=False):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        t1 = time.time()
        with torch.no_grad():
            # switch to evaluate mode
            model.eval()

            end = time.time()
            pbar = tqdm(val_loader, desc='Validation')
            for inputs, targets in pbar:
                # measure data loading time
                data_time.update(time.time() - end)

                input_var, target_var = inputs.to(self.device), targets.to(
                    self.device)

                # compute output
                output = model(input_var)
                loss = self.criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output, target_var, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # update progress bar
                pbar.set_postfix({
                    'Loss': f'{losses.avg:.4f}',
                    'Top1': f'{top1.avg:.4f}',
                    'Top5': f'{top5.avg:.4f}',
                    'Data': f'{data_time.avg:.4f}s',
                    'Batch': f'{batch_time.avg:.4f}s'
                })

        t2 = time.time()
        if verbose:
            logger.info(
                f'* Test loss: {losses.avg:.4f}  top1: {top1.avg:.4f}  top5: {top5.avg:.4f}  time: {t2 - t1:.4f}'
            )
        if self.use_top5:
            return top5.avg
        else:
            return top1.avg
