import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from torch.nn import init
import numpy as np

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

class SACActor(nn.Module):
    # repr_dim will be -1 and feature_dim will be state_dim when we are learning from state
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, from_vision, log_std_bounds, env_name=None):
        super().__init__()
        self.from_vision = from_vision
        self.log_std_bounds = log_std_bounds
        self.env_name = env_name
        if from_vision:
            self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())
        

        action_dim = action_shape[0] * 2
        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_dim))
        self.apply(utils.weight_init)

    def forward(self, obs, bc=False, var=None):
        if self.from_vision:
            h = self.trunk(obs)
        else:
            h = obs
        mu, log_std = self.policy(h).chunk(2, dim=-1)
            

        # constrain log_std inside [log_std_min, log_std_max]
        log_std_min, log_std_max = self.log_std_bounds
        log_std = torch.clip(log_std, log_std_min, log_std_max)
        std_pred = log_std.exp()
        
        if var: log_std = var
        if bc:
            dist = torch.distributions.normal.Normal(mu, std_pred)
        else:
            dist = utils.SquashedNormal(mu, std_pred)

        return dist

class Critic(nn.Module):
    # repr_dim will be -1 and feature_dim will be state_dim when we are learning from state
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, from_vision):
        super().__init__()
        self.from_vision = from_vision

        if from_vision:
            self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        if self.from_vision:
            h = self.trunk(obs)
        else:
            h = obs
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2
