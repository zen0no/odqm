import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=100, n_layers=2):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
        for i in range(n_layers - 2):
            self.net.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.net.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))

    def forward(self, *x):
        assert sum([t.shape[-1] for t in x]) == self.in_dim
        x = torch.cat(x, dim=-1)
        for layer in self.net[:-1]:
            x = F.leaky_relu(layer(x))
        
        return self.net[-1](x)
