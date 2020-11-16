import argparse
import os
import time
from collections import deque

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agent import PrioritizedDoubleDQNAgent
from model import FCModel
from utils import PrioritizedReplayMemory, beta_adder, cur_time

RPM_SIZE = int(1e6)
RPM_WARMUP_SIZE = RPM_SIZE // 20
UPDATE_FREQ = 4


def run_train_episode(env, agent, rpm, get_beta, n_step, gamma, eps):
    obs, done = env.reset(), False
    value_losses = [0.0]
    step, episode_reward = 0, 0
    traj = deque(maxlen=n_step)

    while not done:
        step += 1
        obs = obs.astype('float32')
        if np.random.random() < eps:
            act = env.action_space.sample()
        else:
            act = agent.predict(obs)

        next_obs, reward, done, info = env.step(act)

        traj.append((obs, act, reward, next_obs, done))
        if len(traj) == n_step:
            g = sum(gamma**i * r for i, (_, _, r, _, _) in enumerate(traj))
            rpm.append(obs, act, g, traj[-1][3], traj[-1][4])
        if len(rpm) > RPM_WARMUP_SIZE and step % UPDATE_FREQ == 0:
            batch, weights, idxes = rpm.sample_batch(32, beta=get_beta())
            value_loss, delta = agent.learn(*batch, weights)
            value_losses.append(value_loss)
            rpm.update_priorities(idxes, delta)

        episode_reward += reward

        obs = next_obs
    g = 0
    for i, (obs, act, reward, next_obs, done) in enumerate(reversed(traj)):
        g = reward + gamma * g
        rpm.append(obs, act, g, traj[-1][3], traj[-1][4])

    return episode_reward, np.mean(value_losses)


def run_evaluate_episode(env, agent):
    obs, done = env.reset(), False
    episode_reward = 0
    while not done:
        obs = obs.astype('float32')
        act = agent.predict(obs)
        obs, reward, done, _ = env.step(act)
        episode_reward += reward
    return episode_reward


def run(args):
    torch.manual_seed(args.seed)
    log_path = os.path.join(args.out_log, cur_time())
    writer = SummaryWriter(log_path)

    env = gym.make(args.env)
    rpm = PrioritizedReplayMemory(max_size=RPM_SIZE,
                                  obs_size=env.observation_space.shape,
                                  act_size=(1, ),
                                  reward_size=(1, ),
                                  alpha=args.alpha)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    model = FCModel(obs_dim, act_dim)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = PrioritizedDoubleDQNAgent(model, args, device)

    get_beta = beta_adder(args.beta)

    pbar = tqdm(range(args.num_episodes))
    # Annealing epsilon
    eps = 0.6
    eps_decay_step = (eps - 0.1) / args.num_episodes
    for i in pbar:
        episode_reward, value_loss = run_train_episode(env,
                                                       agent,
                                                       rpm,
                                                       get_beta,
                                                       n_step=args.n_step,
                                                       gamma=args.gamma,
                                                       eps=eps)
        eps = max(0.1, eps - eps_decay_step)
        writer.add_scalar('train_episode_reward', episode_reward, i)
        writer.add_scalar('value_loss', value_loss, i)

        if len(rpm) > RPM_WARMUP_SIZE and (i + 1) % args.eval_freq == 0:
            eval_reward = np.mean(
                [run_evaluate_episode(env, agent) for _ in range(10)])
            writer.add_scalar('evaluate_reward', eval_reward, i)

        if (i + 1) % args.sync_period == 0:
            agent.sync_target()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes',
                        default=10000,
                        type=int,
                        help='number of training episodes')
    parser.add_argument('--env',
                        default='MountainCar-v0',
                        type=str,
                        help='name of the gym environment')
    parser.add_argument('--out_log',
                        default='./logs',
                        type=str,
                        help='path of tensorboard log file')
    parser.add_argument('--seed', default=864, type=int)

    # DQN hyperparameters
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--lr', default=0.00025, type=float)
    parser.add_argument('--eval_freq', default=10, type=int)
    parser.add_argument('--sync_period', default=5, type=int)
    parser.add_argument('--n_step', default=3, type=int)

    # PER hyperparameters
    parser.add_argument('--alpha', default=0.6, type=float)
    parser.add_argument('--beta', default=0.4, type=float)

    args = parser.parse_args()
    run(args)
