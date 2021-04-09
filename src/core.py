import torch
from torch import nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()
        self.action_size = action_size

        self.act = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state, deterministic=False, log_proba=False):
        logits = self.act(state)
        prob = torch.softmax(logits, -1)

        pi_dist = torch.distributions.Categorical(probs=prob)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = pi_dist.sample()

        if log_proba:
            log_prob = F.log_softmax(logits, dim=-1)
        else:
            log_prob = None

        return action, prob, log_prob


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()

        self.approx = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        q_value = self.approx(state)

        return q_value
