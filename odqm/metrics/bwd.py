import torch
import torch.nn as nn
import torch.nn.functional as F

from odqm.network import MLP
from odqm.metrics import BaseMetric

class _Potential(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim=100, **kwargs):
        super().__init__(**kwargs)

        self.net = nn.Sequential(
            nn.Linear(in_features=state_dim + action_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        fc = self.net(x)

        return -torch.abs(fc)


class BellmanWassersteinDistance(BaseMetric):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        qnet_path: str = None,
        max_action: torch.float32 = 1,
        eps: torch.float32 = 1,
        w: torch.float32 = 1,
        **kwargs
    ):
        """

        BellmanWassersteinDistance measures distance between state-action distribution of target data and random state-action distribution
        using algorithm based on optimal transport theory

        Args:
            state_dim (int): State-space dimension
            action_dim (int): Action-space dimension
            critic (nn.Module): Value function for corresponding policy, pretrained on target dataset
            max_action (torch.float32, optional): Maximum of target dataset action distribution. Defaults to 1.
            eps (torch.float32, optional): Coefficient for regularization. Defaults to 1.
            w (torch.float32, optional): Coefficient for dual potentials. Defaults to 1.
        """

        super().__init__(**kwargs)

        self.p1 = _Potential(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.p2 = _Potential(state_dim=state_dim, action_dim=action_dim).to(self.device)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.load_qnet(qnet_path)

        self.eps = eps
        self.w = w

        self.max_action = max_action
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, data):
        state, action = data['state'], data['action']

        action_random = torch.rand_like(action).clamp(-self.max_action, self.max_action)

        p1_vr = self.p1(state, action_random)
        p2_v = self.p2(state, action)

        with torch.no_grad():
            sa_value = -self.q_net(state, action)

        cost = sa_value - F.mse_loss(action, action_random)
        reg = (
            -self.eps
            * torch.exp(1 / self.eps * (p1_vr.flatten() + p2_v.flatten() + cost)).mean()
        )
        loss = -(p1_vr.mean() + self.w * p2_v.mean()) + reg

        return loss

    def load_qnet(self, qnet_path):
        if qnet_path is not None:
            self.q_net = torch.load(qnet_path, map_location=self.device).to(self.device)
        else:
            print("Running dummy qnet")
            self.q_net = MLP(self.state_dim + self.action_dim, 1).to(self.device)

    def train_metric(self, data):
        loss = self(data)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def eval_metric(self, data):
        state, action, reward = data['state'], data['action'], data['reward']
        with torch.no_grad():
            action_random = torch.rand_like(action).clamp(-self.max_action, self.max_action)

            p1_vr = self.p1(state, action_random)
            p2_v = self.p2(state, action)
            distance = p1_vr + p2_v

            output = {
                'bwd': distance.mean().item(),
                'reward': reward.mean().item(),
            }
            return output
