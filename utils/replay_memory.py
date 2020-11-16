import random

import numpy as np

from .segment_tree import MinSegmentTree, SumSegmentTree


class ReplayMemory(object):
    def __init__(self, max_size, obs_size, act_size, reward_size):
        self.max_size = int(max_size)

        self.obs = np.zeros((max_size, ) + obs_size, dtype='float32')
        self.act = np.zeros((max_size, ) + act_size, dtype='int64')
        self.reward = np.zeros((max_size, ) + reward_size, dtype='float32')
        self.next_obs = np.zeros((max_size, ) + obs_size, dtype='float32')
        self.done = np.zeros((max_size, 1), dtype='bool')

        self._cur_size = 0
        self._cur_pos = 0

    def sample_batch(self, batch_size=32):
        idxes = np.random.randint(self._cur_size, size=batch_size)
        obs = self.obs[idxes]
        act = self.act[idxes]
        reward = self.reward[idxes]
        next_obs = self.next_obs[idxes]
        done = self.done[idxes]

        return obs, act, reward, next_obs, done

    def append(self, obs, act, reward, next_obs, done):
        if self._cur_size < self.max_size:
            self._cur_size += 1

        self.obs[self._cur_pos] = obs
        self.act[self._cur_pos] = act
        self.reward[self._cur_pos] = reward
        self.next_obs[self._cur_pos] = next_obs
        self.done[self._cur_pos] = done

        self._cur_pos = (self._cur_pos + 1) % self.max_size

    def __len__(self):
        return self._cur_size


class PrioritizedReplayMemory(ReplayMemory):
    def __init__(self, max_size, obs_size, act_size, reward_size, alpha):
        super().__init__(max_size, obs_size, act_size, reward_size)
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < max_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def append(self, *args, **kwargs):
        idx = self._cur_pos
        super().append(*args, **kwargs)
        self._it_sum[idx] = self._max_priority**self._alpha
        self._it_min[idx] = self._max_priority**self._alpha

    def _sample_proportional(self, batch_size):
        idxes = []
        p_total = self._it_sum.sum(0, self._cur_size - 1)
        range_step = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * range_step + i * range_step
            idx = self._it_sum.find_prefixsum_idx(mass)
            idxes.append(idx)
        return idxes

    def sample_batch(self, batch_size, beta):
        assert beta > 0
        idxes = self._sample_proportional(batch_size)

        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self._cur_size)**(-beta)

        weights = []
        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self._cur_size)**(-beta)
            weights.append(weight / max_weight)

        weights = np.array(weights)
        obs = self.obs[idxes]
        act = self.act[idxes]
        reward = self.reward[idxes]
        next_obs = self.next_obs[idxes]
        done = self.done[idxes]
        return (obs, act, reward, next_obs, done), weights, idxes

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        priorities = priorities + 0.001
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self._cur_size
            self._it_sum[idx] = priority**self._alpha
            self._it_min[idx] = priority**self._alpha

            self._max_priority = max(self._max_priority, priority)
