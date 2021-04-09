import torch


class GumbelSoftmax(torch.distributions.RelaxedOneHotCategorical):
    def sample(self, sample_shape=torch.Size()):
        '''Gumbel-softmax sampling. Note rsample is inherited from RelaxedOneHotCategorical'''
        u = torch.empty(self.logits.size(), device=self.logits.device, dtype=self.logits.dtype).uniform_(0, 1)
        noisy_logits = self.logits - torch.log(-torch.log(u))
        return torch.argmax(noisy_logits, dim=-1)

    def rsample(self, sample_shape=torch.Size()):
        '''
        Gumbel-softmax resampling using the Straight-Through trick.
        Credit to Ian Temple for bringing this to our attention. To see standalone code of how this works, refer to https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
        '''
        rout = super().rsample(sample_shape)  # differentiable
        out = F.one_hot(torch.argmax(rout, dim=-1), self.logits.shape[-1]).float()
        return (out - rout).detach() + rout

    def log_prob(self, value):
        '''value is one-hot or relaxed'''
        if value.shape != self.logits.shape:
            value = F.one_hot(value.long(), self.logits.shape[-1]).float()
            assert value.shape == self.logits.shape
        return - torch.sum(- value * F.log_softmax(self.logits, -1), -1)


class Actor(nn.Module):
    def __init__(self, state_size, action_size, temp, hidden=256):
        super().__init__()
        self.temp = temp

        self.act = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, action_size),
        )

    def forward(self, state, reparam=False):
        logits = self.act(state)

        gumbel_dist = GumbelSoftmax(self.temp, logits=logits)

        if reparam:
            action = gumbel_dist.rsample()
        else:
            action = gumbel_dist.sample()

        log_pi = gumbel_dist.log_prob(action)

        return action, log_pi


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden=256):
        super().__init__()

        self.approx = nn.Sequential(
            nn.Linear(state_size + action_size, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        q_value = self.approx(x)

        return q_value.flatten()


class SACGumbel:
    def __init__(self, state_size, action_size,
                 config, device='cpu', dtype=torch.float32):
        self.gamma = config.gamma
        self.tau = config.tau
        self.dtype = dtype
        self.device = device
        self.action_size = action_size
        self.hidden = config.hidden
        self.clip_grad = config.clip_grad

        self.actor = Actor(state_size, action_size, temp=config.temp, hidden=self.hidden)
        self.critic_1 = Critic(state_size, action_size, self.hidden)
        self.critic_2 = Critic(state_size, action_size, self.hidden)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        self.actor.to(device)
        self.critic_1.to(device)
        self.target_critic_1.to(device)
        self.critic_2.to(device)
        self.target_critic_2.to(device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt_1 = torch.optim.Adam(self.critic_1.parameters(), lr=config.critic_lr)
        self.critic_opt_2 = torch.optim.Adam(self.critic_2.parameters(), lr=config.critic_lr)

        for p in itertools.chain(self.target_critic_1.parameters(), self.target_critic_2.parameters()):
            p.requires_grad = False

        # adaptive temperature
        # self.target_entropy = -np.log((1.0 / self.action_size)) * config.target_entropy_scale
        self.target_entropy = -4
        self.init_alpha = np.log(config.alpha)

        self.alpha_log = torch.tensor([self.init_alpha], dtype=dtype,
                                      device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.alpha_log], lr=config.alpha_lr)
        self.alpha = torch.exp(self.alpha_log)

    def one_hot(self, actions):
        actions = F.one_hot(actions.long(), self.action_size).float()

        return actions

    def _compute_alpha(self, log_pi):
        loss = - torch.mean((self.alpha_log * (log_pi.detach() + self.target_entropy)))

        return loss

    def _compute_target(self, reward, next_state, done):
        with torch.no_grad():
            action, log_pi = self.actor(next_state, reparam=False)
            action = self.one_hot(action)

            assert action.size(-1) == self.action_size, 'Dimensions are wrong!'

            q1_targ = self.target_critic_1(next_state, action)
            q2_targ = self.target_critic_2(next_state, action)
            q_min = torch.min(q1_targ, q2_targ)

            target = reward + self.gamma * (1 - done) * (q_min - self.alpha * log_pi)

        return target

    def _compute_actor_loss(self, state):
        """
        Принимает на вход state, который является объектом класса torch.Tensor и имеет размерность (batch_size, state_size)
        Возвращает функцию потерь actor'а loss, которая явлеятся объектом класса torch.Tensor
        """
        action, log_pi = self.actor(state, reparam=True)

        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)
        q_min = torch.min(q1, q2)

        loss = torch.mean(q_min - self.alpha * log_pi)

        return -1 * loss, log_pi

    def _compute_critic_loss(self, state, action, next_state, reward, done):
        target = self._compute_target(reward, next_state, done)
        action = self.one_hot(action)  # do not comment this line idiot!

        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)

        assert q1.size() == target.size()
        assert q2.size() == target.size()

        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        return loss

    def _soft_update(self, target_net, source_net):
        """
        Применяет soft update с коэффициентом self.tau обновляя параметры target_net с помощью параметров source_net
        """
        for tp, lp in zip(target_net.parameters(), source_net.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * lp.data)

    def _grad_clip(self, net):
        nn.utils.clip_grad_norm_(net.parameters(), self.clip_grad)

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

        if self.clip_grad is not None:
            self._grad_clip(self.critic_1)
            self._grad_clip(self.critic_2)

        self.critic_opt_1.step()
        self.critic_opt_2.step()

        actor_loss, log_pi = self._compute_actor_loss(state)
        self.actor_optim.zero_grad()
        actor_loss.backward()

        if self.clip_grad is not None:
            self._grad_clip(self.actor)

        self.actor_optim.step()

        alpha_loss = self._compute_alpha(log_pi)
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

    def get_alpha(self):
        return self.alpha.item()

    def get_action(self, state, det=False, reparam=False):
        """
        Принимает на вход state, который является объектом класса numpy.array
        Возвращает action, который является объектом класса numpy.array
        """
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=self.dtype, device=self.device)
            action, _ = self.actor(state, det, reparam)

        if action.dim() > 0:
            if action.size(-1) != 1:
                action = torch.argmax(action, -1)

        return action.item()