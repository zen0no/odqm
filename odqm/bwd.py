import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BellmanWassersteinDistance"]


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
        x = torch.cat([state, action])
        fc = self.forward(x)

        return -torch.abs(fc)


class BellmanWassersteinDistance(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        value_func: nn.Module,
        max_action: torch.float32 = 1,
        eps: torch.float32 = 1,
        W: torch.float32 = 1,
        device: str = None,
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
            W (torch.float32, optional): Coefficient for dual potentials. Defaults to 1.
            device (str, optional): Device, where distance will be trained. Defaults to None.
        """
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        super().__init__()

        self.p1 = _Potential(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.p2 = _Potential(state_dim=state_dim, action_dim=action_dim).to(self.device)

        self.value_func = value_func.to(self.device)

        self.eps = eps
        self.W = W

        self.max_action = max_action

    def forward(self, state, action):
        action_random = torch.rand_like(action).clamp(-self.max_action, self.max_action)

        p1_v = self.p1(state, action)
        p2_v = self.p2(state, action)
        p2_vr = self.p2(state, action_random)

        with torch.inference_mode():
            sa_value = -self.value_func(state, action)

        cost = sa_value - F.mse_loss(action, action_random)
        reg = (
            -self.eps
            * torch.exp(1 / self.eps * (p1_v.flatten() + p2_vr.flatten() + cost)).mean()
        )
        loss = -(p1_v.mean() + self.W * p2_v.mean()) + reg

        return loss
