from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DoubleDQNAgent(object):
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.target_model = deepcopy(self.model).to(device)
        self.opt = optim.Adam(params=self.model.parameters(), lr=config.lr)

        self.gamma = config.gamma

        self.device = device

    def predict(self, obs):
        obs = torch.from_numpy(obs).to(self.device)
        out = self.model.policy(obs)
        return np.squeeze(out.cpu().detach().numpy())

    def learn(self, obs, act, reward, next_obs, done):
        obs = torch.from_numpy(obs).to(self.device)
        act = torch.from_numpy(act).to(self.device)
        reward = torch.from_numpy(reward).to(self.device)
        next_obs = torch.from_numpy(next_obs).to(self.device)
        done = torch.from_numpy(done).to(self.device)

        loss = self._update_value(obs, act, reward, next_obs, done)
        return loss.cpu().detach().numpy()

    def _update_value(self, obs, act, reward, next_obs, done):
        with torch.no_grad():
            next_greedy_action = self.model.policy(next_obs)
            next_greedy_action = torch.unsqueeze(next_greedy_action, axis=1)
            target_value = self.target_model.value(next_obs)
            target_value = target_value.gather(1, next_greedy_action)
            target_Q = reward + \
                self.gamma * (1.0 - done.float()) * target_value

        value = self.model.value(obs)
        Q = value.gather(1, act)
        loss = F.mse_loss(Q, target_Q)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss

    def sync_target(self):
        for target_param, param in zip(self.target_model.parameters(),
                                       self.model.parameters()):
            target_param.data.copy_(param.data)


class PrioritizedDoubleDQNAgent(object):
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.target_model = deepcopy(self.model).to(device)
        self.opt = optim.Adam(params=self.model.parameters(), lr=config.lr)

        self.gamma = config.gamma

        self.device = device

    def predict(self, obs):
        obs = torch.from_numpy(obs).to(self.device)
        out = self.model.policy(obs)
        return np.squeeze(out.cpu().detach().numpy())

    def learn(self, obs, act, reward, next_obs, done, weights):
        obs = torch.from_numpy(obs).to(self.device)
        act = torch.from_numpy(act).to(self.device)
        reward = torch.from_numpy(reward).to(self.device)
        next_obs = torch.from_numpy(next_obs).to(self.device)
        done = torch.from_numpy(done).to(self.device)
        weights = torch.from_numpy(weights).to(self.device)

        loss, delta = self._update_value(obs, act, reward, next_obs, done,
                                         weights)
        return loss.cpu().detach().numpy(), delta.cpu().detach().numpy()

    def _update_value(self, obs, act, reward, next_obs, done, weights):
        with torch.no_grad():
            next_greedy_action = self.model.policy(next_obs)
            next_greedy_action = torch.unsqueeze(next_greedy_action, axis=1)
            target_value = self.target_model.value(next_obs)
            target_value = target_value.gather(1, next_greedy_action)
            target_Q = reward + \
                self.gamma * (1.0 - done.float()) * target_value

        value = self.model.value(obs)
        Q = value.gather(1, act)
        delta = torch.abs(target_Q - Q)
        loss = weights * F.mse_loss(Q, target_Q, reduction='none')
        loss = torch.mean(loss)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss, delta

    def sync_target(self):
        for target_param, param in zip(self.target_model.parameters(),
                                       self.model.parameters()):
            target_param.data.copy_(param.data)
