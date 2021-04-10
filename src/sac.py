import torch
import torch.nn.functional as F
import copy
import itertools
import numpy as np
from src.core import Actor, Critic


class SACDiscrete:
    def __init__(self, config, dtype=torch.float32):
        self.state_size = config['state_size']
        self.action_size = config['action_size']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.dtype = dtype
        self.device = config['device']
        self.hidden = config['hidden']

        self.actor = Actor(self.state_size, self.action_size, hidden_size=self.hidden)
        self.critic_1 = Critic(self.state_size, self.action_size, hidden_size=self.hidden)
        self.critic_2 = Critic(self.state_size, self.action_size, hidden_size=self.hidden)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        self.actor.to(self.device)
        self.critic_1.to(self.device)
        self.target_critic_1.to(self.device)
        self.critic_2.to(self.device)
        self.target_critic_2.to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.critic_opt_1 = torch.optim.Adam(self.critic_1.parameters(), lr=config['critic_lr'])
        self.critic_opt_2 = torch.optim.Adam(self.critic_2.parameters(), lr=config['critic_lr'])

        for p in itertools.chain(self.target_critic_1.parameters(), self.target_critic_2.parameters()):
            p.requires_grad = False

        # adaptive temperature
        self.target_entropy = -np.log((1.0 / self.action_size)) * config['target_entropy_scale']
        self.init_alpha = np.log(config['alpha'])

        self.alpha_log = torch.tensor([self.init_alpha], dtype=self.dtype,
                                      device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.alpha_log], lr=config['alpha_lr'])
        self.alpha = torch.exp(self.alpha_log)

    def _compute_alpha(self, state):
        action, prob, log_prob = self.actor(state, deterministic=False, log_proba=True)

        loss = torch.mean((-1 * self.alpha_log * (torch.sum(log_prob * prob, -1) + self.target_entropy)))

        return loss

    def _compute_target(self, reward, next_state, done):
        with torch.no_grad():
            a_prime, prob, log_prob = self.actor(next_state, deterministic=False, log_proba=True)

            q1_targ = self.target_critic_1(next_state)
            q2_targ = self.target_critic_2(next_state)
            q_min = torch.min(q1_targ, q2_targ)

            q = torch.sum((q_min - self.alpha * log_prob) * prob, -1)
            target = reward + self.gamma * (1 - done) * q

        return target

    def _compute_actor_loss(self, state):
        action, prob, log_prob = self.actor(state, deterministic=False, log_proba=True)

        q1 = self.critic_1(state)
        q2 = self.critic_2(state)
        q_min = torch.min(q1, q2)

        loss = torch.mean(torch.sum((q_min - self.alpha.detach() * log_prob) * prob, -1))

        return -1 * loss

    def _compute_critic_loss(self, state, action, next_state, reward, done):
        target = self._compute_target(reward, next_state, done)

        q1 = self.critic_1(state).gather(1, action.reshape(-1, 1).long()).view(-1)
        q2 = self.critic_2(state).gather(1, action.reshape(-1, 1).long()).view(-1)

        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        return loss

    def _soft_update(self, target_net, source_net):
        for tp, lp in zip(target_net.parameters(), source_net.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * lp.data)

    def update(self, batch):
        state, action, reward, next_state, done = batch

        state = torch.tensor(state, dtype=self.dtype, device=self.device)
        next_state = torch.tensor(next_state, dtype=self.dtype, device=self.device)
        reward = torch.tensor(reward, dtype=self.dtype, device=self.device)
        done = torch.tensor(done, device=self.device, dtype=self.dtype)
        action = torch.tensor(action, device=self.device, dtype=self.dtype)

        critic_loss = self._compute_critic_loss(state, action, next_state, reward, done)
        self.critic_opt_1.zero_grad()
        self.critic_opt_2.zero_grad()
        critic_loss.backward()
        self.critic_opt_1.step()
        self.critic_opt_2.step()

        actor_loss = self._compute_actor_loss(state)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = self._compute_alpha(state)
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = torch.exp(self.alpha_log)

        # Critic soft update
        with torch.no_grad():
            self._soft_update(self.target_critic_1, self.critic_1)
            self._soft_update(self.target_critic_2, self.critic_2)

        return actor_loss.item(), critic_loss.item()

    def save_model(self, path):
        torch.save(self.actor.state_dict(), f'{path}/actor.pt')
        torch.save(self.critic_1.state_dict(), f'{path}/critic_1.pt')
        torch.save(self.critic_2.state_dict(), f'{path}/critic_2.pt')

    def load_model(self, path_act, path_crit_1, path_crit_2):
        self.actor.load_state_dict(torch.load(path_act, map_location=self.device))
        self.critic_1.load_state_dict(torch.load(path_crit_1, map_location=self.device))
        self.critic_2.load_state_dict(torch.load(path_crit_2, map_location=self.device))

        print('Weights are loaded')

    def get_alpha(self):
        return self.alpha.item()

    def get_action(self, state, deterministic=False, log_proba=False):
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=self.dtype, device=self.device)
            action, _, _ = self.actor(state, deterministic, log_proba)

        return action.item()