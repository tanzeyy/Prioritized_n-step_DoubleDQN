import torch
import torch.nn as nn
import torch.nn.functional as F


class FCModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.value = FCVAlue(obs_dim, act_dim)

    def policy(self, obs):
        with torch.no_grad():
            Q = self.value(obs)
            return torch.argmax(Q, axis=-1)


class FCVAlue(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, act_dim)

    def forward(self, obs):
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
