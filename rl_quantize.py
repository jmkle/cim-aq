# Code for "[HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
# Kuan Wang*, Zhijian Liu*, Yujun Lin*, Ji Lin, Song Han
# {kuanwang, zhijian, yujunlin, jilin, songhan}@mit.edu

import argparse
import math
import os
from copy import deepcopy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import models as customized_models
from lib.env.linear_quantize_env import LinearQuantizeEnv
from lib.env.quantize_env import QuantizeEnv
from lib.rl.ddpg import DDPG
from lib.utils.logger import logger

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


def train(num_episode, agent, env, output, debug=False, wandb_enable=False):
    # best record
    best_reward = -math.inf
    best_policy = []
    best_accuracy = 0.0

    # Timing and progress tracking
    start_time = time.time()
    episode_times = []

    # Statistics tracking
    reward_history = []
    accuracy_history = []
    cost_history = []

    # Create progress bar
    pbar = tqdm(total=num_episode,
                desc="RL Quantization",
                unit="ep",
                dynamic_ncols=True,
                leave=True)

    # Initial progress bar postfix
    pbar.set_postfix({
        'Best_Rew': f'{best_reward:.4f}',
        'Best_Acc': f'{best_accuracy:.4f}%',
        'Orig_Acc': f'{env.org_acc:.4f}%',
        'Target': f'{env.compress_ratio:.4f}'
    })

    # Periodic logging intervals
    log_interval = max(1, num_episode // 20)  # Log 20 times during training
    checkpoint_interval = max(1, num_episode // 10)  # Save 10 checkpoints

    logger.info(
        f"Starting RL-based quantization search for {num_episode} episodes")
    logger.info(f"Target preserve ratio: {env.compress_ratio:.4f}")
    logger.info(f"Original model accuracy: {env.org_acc:.4f}%")
    if hasattr(env, 'target_acc') and env.target_acc is not None:
        logger.info(f"Target accuracy constraint: {env.target_acc:.4f}%")

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    T = []  # trajectory

    while episode < num_episode:  # counting based on episode
        episode_start_time = time.time()

        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if episode <= args.warmup:
            action = agent.random_action()
            action_type = "random"
        else:
            action = agent.select_action(observation, episode=episode)
            action_type = "policy"

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)

        T.append([
            reward,
            deepcopy(observation),
            deepcopy(observation2), action, done
        ])

        # [optional] save intermideate model
        if episode % checkpoint_interval == 0 and episode > 0:
            agent.save_model(output)
            logger.info(f"Checkpoint saved at episode {episode}")

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done:  # end of episode
            episode_end_time = time.time()
            episode_duration = episode_end_time - episode_start_time
            episode_times.append(episode_duration)

            # Track statistics
            accuracy = info.get('accuracy', 0.0)
            cost = info.get('cost', 0.0)

            reward_history.append(episode_reward)
            accuracy_history.append(accuracy)
            cost_history.append(cost)

            # Calculate timing metrics for progress bar
            elapsed_time = time.time() - start_time
            avg_episode_time = np.mean(
                episode_times[-10:]) if len(episode_times) >= 10 else np.mean(
                    episode_times) if episode_times else 0
            remaining_episodes = num_episode - episode - 1
            eta_seconds = avg_episode_time * remaining_episodes

            # Calculate recent statistics
            recent_window = min(10, len(reward_history))
            recent_rewards = reward_history[
                -recent_window:] if recent_window > 0 else [episode_reward]
            recent_accs = accuracy_history[
                -recent_window:] if recent_window > 0 else [accuracy]

            # Update progress bar
            pbar.update(1)

            # Create detailed postfix information
            postfix_dict = {
                'Rew': f'{episode_reward:.4f}',
                'Acc': f'{accuracy:.4f}%',
                'Best_R': f'{best_reward:.4f}',
                'Best_A': f'{best_accuracy:.4f}%',
                'Avg_R': f'{np.mean(recent_rewards):.4f}',
                'Avg_A': f'{np.mean(recent_accs):.4f}%'
            }

            cost_display = cost * 1. / 8e6
            cost_ratio = info.get('cost_ratio', 0.0)
            postfix_dict['Cost'] = f'{cost_display:.4f}'
            postfix_dict['C_Ratio'] = f'{cost_ratio:.4f}'

            pbar.set_postfix(postfix_dict)

            cost_display = cost * 1. / 8e6
            cost_ratio = info.get('cost_ratio', 0.0)
            if debug or episode % log_interval == 0:
                logger.info(
                    f'Episode {episode:4d}: reward={episode_reward:7.4f}, '
                    f'acc={accuracy:6.4f}%, cost={cost_display:7.4f}, '
                    f'cost_ratio={cost_ratio:.4f}, action={action_type}, '
                    f'time={episode_duration:.4f}s')
            text_writer.write(
                '#{}: episode_reward:{:.4f} acc: {:.4f}, cost: {:.4f}\n'.
                format(episode, episode_reward, accuracy, cost_display))

            final_reward = T[-1][0]

            # Update best policy tracking
            if final_reward > best_reward:
                best_reward = final_reward
                best_policy = env.quantization_strategy.copy()
                best_accuracy = accuracy
                logger.info(
                    f"*** NEW BEST POLICY found at episode {episode} ***")
                logger.info(f"    Best reward: {best_reward:.4f}")
                logger.info(f"    Best accuracy: {best_accuracy:.4f}%")
                logger.info(f"    Strategy: {best_policy}")

                # Save best policy immediately
                np.save(os.path.join(output, 'best_policy.npy'), best_policy)

            # agent observe and update policy
            for i, (r_t, s_t, s_t1, a_t, done) in enumerate(T):
                agent.observe(final_reward, s_t, s_t1, a_t, done)
                if episode > args.warmup:
                    for i in range(args.n_update):
                        agent.update_policy()

            agent.memory.append(
                observation, agent.select_action(observation, episode=episode),
                0., False)

            # Progress reporting and ETA calculation
            if episode % log_interval == 0 and episode > 0:
                # Write detailed progress to log (less frequent than progress bar)
                logger.info(
                    f"=== Detailed Progress - Episode {episode}/{num_episode} ==="
                )
                logger.info(
                    f"Elapsed time: {str(timedelta(seconds=int(elapsed_time)))}"
                )
                logger.info(
                    f"Estimated time remaining: {str(timedelta(seconds=int(eta_seconds)))}"
                )
                logger.info(f"Average episode time: {avg_episode_time:.4f}s")
                logger.info(f"Episodes per second: {1/avg_episode_time:.4f}")
                logger.info(
                    f"Recent avg reward (last {recent_window}): {np.mean(recent_rewards):.4f}"
                )
                logger.info(
                    f"Recent avg accuracy (last {recent_window}): {np.mean(recent_accs):.4f}%"
                )
                logger.info(f"Best reward so far: {best_reward:.4f}")
                logger.info(f"Best accuracy so far: {best_accuracy:.4f}%")
                logger.info(f"Progress: {episode/num_episode*100:.1f}%")
                logger.info("=" * 50)

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            T = []

            value_loss = agent.get_value_loss()
            policy_loss = agent.get_policy_loss()
            delta = agent.get_delta()
            tfwriter.add_scalar('reward/last', final_reward, episode)
            tfwriter.add_scalar('reward/best', best_reward, episode)
            tfwriter.add_scalar('info/accuracy', info['accuracy'], episode)
            tfwriter.add_text('info/best_policy', str(best_policy), episode)
            tfwriter.add_text('info/current_policy', str(env.strategy),
                              episode)
            tfwriter.add_scalar('value_loss', value_loss, episode)
            tfwriter.add_scalar('policy_loss', policy_loss, episode)
            tfwriter.add_scalar('delta', delta, episode)

            tfwriter.add_scalar('info/cost_ratio', info.get('cost_ratio', 0.0),
                                episode)
            # record the preserve rate for each layer
            for i, preserve_rate in enumerate(env.strategy):
                tfwriter.add_scalar('preserve_rate_w/{}'.format(i),
                                    preserve_rate[0], episode)
                tfwriter.add_scalar('preserve_rate_a/{}'.format(i),
                                    preserve_rate[1], episode)

            # ============ W&B logging ============#
            if wandb_enable:
                wandb_metrics = {
                    'reward/last': final_reward,
                    'reward/best': best_reward,
                    'info/accuracy': info['accuracy'],
                    'value_loss': value_loss,
                    'policy_loss': policy_loss,
                    'delta': delta,
                    'info/cost_ratio': info.get('cost_ratio', 0.0),
                    'info/best_policy': str(best_policy),
                    'info/current_policy': str(env.strategy)
                }

                # Add per-layer preserve rates
                for i, preserve_rate in enumerate(env.strategy):
                    wandb_metrics[f'preserve_rate_w/{i}'] = preserve_rate[0]
                    wandb_metrics[f'preserve_rate_a/{i}'] = preserve_rate[1]

                wandb.log(wandb_metrics, step=episode)

            text_writer.write('best reward: {}\n'.format(best_reward))
            text_writer.write('best policy: {}\n'.format(best_policy))

    # Close progress bar
    pbar.close()

    # Final summary
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("RL QUANTIZATION SEARCH COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total episodes: {num_episode}")
    logger.info(f"Total time: {str(timedelta(seconds=int(total_time)))}")
    logger.info(f"Average time per episode: {total_time/num_episode:.4f}s")
    logger.info(f"Final best reward: {best_reward:.4f}")
    logger.info(f"Final best accuracy: {best_accuracy:.4f}%")
    logger.info(f"Original accuracy: {env.org_acc:.4f}%")
    accuracy_drop = env.org_acc - best_accuracy
    logger.info(f"Accuracy drop: {accuracy_drop:.4f}%")
    logger.info(f"Best quantization strategy: {best_policy}")

    # Save final statistics
    stats = {
        'total_episodes': num_episode,
        'total_time': total_time,
        'best_reward': best_reward,
        'best_accuracy': best_accuracy,
        'original_accuracy': env.org_acc,
        'accuracy_drop': accuracy_drop,
        'best_policy': best_policy,
        'reward_history': reward_history,
        'accuracy_history': accuracy_history,
        'cost_history': cost_history
    }
    with open(os.path.join(output, 'training_stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)
    logger.info(f"Training statistics saved to {output}/training_stats.pkl")

    text_writer.close()
    return best_policy, best_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PyTorch Reinforcement Learning')

    parser.add_argument(
        '--suffix',
        default=None,
        type=str,
        help='suffix to help you remember what experiment you ran')
    # env
    parser.add_argument('--dataset',
                        default='imagenet',
                        type=str,
                        help='dataset to use')
    parser.add_argument('--dataset_root',
                        default='data/imagenet',
                        type=str,
                        help='path to dataset')
    parser.add_argument('--preserve_ratio',
                        default=0.1,
                        type=float,
                        help='preserve ratio of the model size')
    parser.add_argument('--min_bit',
                        default=1,
                        type=float,
                        help='minimum bit to use')
    parser.add_argument('--max_bit',
                        default=8,
                        type=float,
                        help='maximum bit to use')
    parser.add_argument('--float_bit',
                        default=32,
                        type=int,
                        help='the bit of full precision float')
    parser.add_argument('--linear_quantization',
                        dest='linear_quantization',
                        action='store_true')
    parser.add_argument('--is_pruned', dest='is_pruned', action='store_true')
    # ddpg
    parser.add_argument('--hidden1',
                        default=300,
                        type=int,
                        help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2',
                        default=300,
                        type=int,
                        help='hidden num of second fully connect layer')
    parser.add_argument('--lr_c',
                        default=1e-3,
                        type=float,
                        help='learning rate for actor')
    parser.add_argument('--lr_a',
                        default=1e-4,
                        type=float,
                        help='learning rate for actor')
    parser.add_argument(
        '--warmup',
        default=20,
        type=int,
        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=1., type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize',
                        default=128,
                        type=int,
                        help='memory size for each layer')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau',
                        default=0.01,
                        type=float,
                        help='moving average for target network')
    # noise (truncated normal distribution)
    parser.add_argument(
        '--init_delta',
        default=0.5,
        type=float,
        help='initial variance of truncated normal distribution')
    parser.add_argument('--delta_decay',
                        default=0.99,
                        type=float,
                        help='delta decay during exploration')
    parser.add_argument('--n_update',
                        default=1,
                        type=int,
                        help='number of rl to update each time')
    # training
    parser.add_argument('--max_episode_length', default=1e9, type=int, help='')
    parser.add_argument('--output',
                        default=os.path.abspath(
                            os.path.join(os.path.dirname(__file__), 'save')),
                        type=str,
                        help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_episode',
                        default=600,
                        type=int,
                        help='train iters each timestep')
    parser.add_argument('--epsilon',
                        default=50000,
                        type=int,
                        help='linear decay of exploration policy')
    parser.add_argument('--seed', default=234, type=int, help='')
    parser.add_argument('--n_worker',
                        default=32,
                        type=int,
                        help='number of data loader worker')
    parser.add_argument('--data_bsize',
                        default=256,
                        type=int,
                        help='number of data batch size')
    parser.add_argument('--finetune_epoch', default=1, type=int, help='')
    parser.add_argument('--finetune_gamma',
                        default=0.8,
                        type=float,
                        help='finetune gamma')
    parser.add_argument('--finetune_lr',
                        default=0.001,
                        type=float,
                        help='finetune gamma')
    parser.add_argument('--finetune_flag',
                        default=True,
                        type=bool,
                        help='whether to finetune')
    parser.add_argument('--use_top5',
                        default=False,
                        type=bool,
                        help='whether to use top5 acc in reward')
    parser.add_argument('--train_size',
                        default=20000,
                        type=int,
                        help='number of train data size')
    parser.add_argument('--val_size',
                        default=10000,
                        type=int,
                        help='number of val data size')
    parser.add_argument('--resume',
                        default='default',
                        type=str,
                        help='Resuming model path for testing')
    parser.add_argument('--amp', action='store_true', help='use amp')
    # Architecture
    parser.add_argument('--arch',
                        '-a',
                        metavar='ARCH',
                        default='mobilenet_v2',
                        choices=model_names,
                        help='model architecture:' + ' | '.join(model_names) +
                        ' (default: mobilenet_v2)')
    # device options
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
    base_folder_name = '{}_{}'.format(args.arch, args.dataset)
    if args.suffix is not None:
        base_folder_name = base_folder_name + '_' + args.suffix
    args.output = os.path.join(args.output, base_folder_name)

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    tfwriter = SummaryWriter(log_dir=args.output)
    text_writer = open(os.path.join(args.output, 'log.txt'), 'w')

    # Initialize W&B if enabled
    if args.wandb_enable:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"rl_quantize_{base_folder_name}",
            config=vars(args),
            tags=['rl_quantization', args.arch, args.dataset])

    logger.info('==> Output path: {}...'.format(args.output))

    # Use CUDA if available, otherwise use CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'==> Using device: {device}')
    logger.info(f'==> GPU IDs: {args.gpu_id}')

    if args.seed > 0:
        logger.info(f'==> Setting random seed: {args.seed}')
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed)

    if args.dataset == 'imagenet':
        num_classes = 1000
    elif args.dataset == 'imagenet100':
        num_classes = 100
    else:
        raise NotImplementedError
    model = models.__dict__[args.arch](pretrained=True,
                                       num_classes=num_classes)
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.to(device)
    else:
        model = torch.nn.DataParallel(model).to(device)
    pretrained_model = deepcopy(model.state_dict())
    logger.info('    Total params: %.2fM' %
                (sum(p.numel() for p in model.parameters()) / 1000000.0))
    cudnn.benchmark = True

    logger.info(f'==> Loading model: {args.arch}')
    logger.info(f'==> Dataset: {args.dataset} (classes: {num_classes})')
    logger.info(f'==> Dataset root: {args.dataset_root}')

    logger.info('==> Initializing quantization environment...')
    logger.info(f'==> Preserve ratio: {args.preserve_ratio}')
    logger.info(f'==> Min bit: {args.min_bit}, Max bit: {args.max_bit}')

    if args.linear_quantization:
        env = LinearQuantizeEnv(model,
                                pretrained_model,
                                args.dataset,
                                args.dataset_root,
                                compress_ratio=args.preserve_ratio,
                                n_data_worker=args.n_worker,
                                batch_size=args.data_bsize,
                                args=args,
                                float_bit=args.float_bit,
                                is_model_pruned=args.is_pruned)
    else:
        env = QuantizeEnv(model,
                          pretrained_model,
                          args.dataset,
                          args.dataset_root,
                          compress_ratio=args.preserve_ratio,
                          n_data_worker=args.n_worker,
                          batch_size=args.data_bsize,
                          args=args,
                          float_bit=args.float_bit,
                          is_model_pruned=args.is_pruned)

    nb_states = env.layer_embedding.shape[1]
    nb_actions = 1  # actions for weight and activation quantization
    args.rmsize = args.rmsize * len(env.quantizable_idx)  # for each layer

    logger.info(
        f'==> Number of quantizable layers: {len(env.quantizable_idx)}')
    logger.info(f'==> State embedding shape: {env.layer_embedding.shape}')
    logger.info(f'==> Actual replay buffer size: {args.rmsize}')

    logger.info('==> Initializing DDPG agent...')
    agent = DDPG(nb_states, nb_actions, args)
    agent.to(device)

    best_policy, best_reward = train(args.train_episode,
                                     agent,
                                     env,
                                     args.output,
                                     debug=args.debug,
                                     wandb_enable=args.wandb_enable)
    logger.info(f'best_reward: {best_reward}')
    logger.info(f'best_policy: {best_policy}')

    # save the best policy
    np.save(os.path.join(args.output, 'best_policy.npy'), best_policy)
