import torch
import gym
import numpy as np

from collections import Tuple


def buffer_from_gym(env, **kwargs):
    pass

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.size = max_size

        self.states = np.zeros((state_dim, max_size))
        self.actions = np.array((action_dim, max_size))
        self.next_states = np.array((state_dim, max_size))
        self.next_actions = np.array((action_dim, max_size))

        self.rewards = np.zeros((max_size, ))

    def sample(self, batch_size: int) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        idx = np.random.randint()

        return (
            torch.FloatTensor(self.states[idx], device=self.device),
            torch.FloatTensor(self.actions[idx], device=self.device),
            torch.FloatTensor(self.newstates[idx], device=self.device),
            torch.FloatTensor(self.new_actions[idx], device=self.device),
        )