import torch
import torch.nn as nn
import numpy as np


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    """Build a feedforward neural network."""
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class RewardNetwork(nn.Module):
    """Reward model for CartPole states."""
    def __init__(self):
        super().__init__()
        self.reward_net = mlp(sizes=[4, 64, 64, 1], activation=nn.Tanh, output_activation=nn.Identity)

    def forward(self, x):
        return self.reward_net(x)

    def predict_reward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
            return self.forward(state).squeeze(0).squeeze(-1)
        return self.forward(state).squeeze(-1)

    def predict_return(self, traj):
        """Estimate the cumulative reward for a trajectory."""
        if not isinstance(traj, torch.Tensor):
            traj = torch.as_tensor(np.asarray(traj), dtype=torch.float32)
        traj = traj.to(next(self.parameters()).device)
        return self.predict_reward(traj).sum()