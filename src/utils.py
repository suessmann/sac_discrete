from collections import deque
import wandb
import matplotlib.pyplot as plt
import gym
import numpy as np
import random


class ExperienceBuffer:
    def __init__(self, buff_len, batch_size):
        self.buff = deque(maxlen=buff_len)
        self.batch_size = batch_size

    def add(self, observation):
        self.buff.append(observation)

    def sample(self):
        out = random.sample(self.buff, self.batch_size)
        return list(zip(*out))  # transpose list

    def clear(self):
        self.buff.clear()

    def __len__(self):
        return len(self.buff)


def train(model, config):
    env = gym.make(config['env'])
    env.action_space.seed(42)
    env.seed(42)

    state = env.reset()

    done = 0
    total_reward = 0
    reward_old = 0

    loss_act_hist = []
    loss_crit_hist = []
    reward_mean_hist = []
    reward_std_hist = []
    timestep_hist = []

    buffer = ExperienceBuffer(config['mem_size'], config['batch_size'])

    for it in np.arange(config['timesteps']):
        if done:
            alpha = model.get_alpha()
            wandb.log({'reward_train': total_reward, 'timestep': it})
            wandb.log({'loss_act': np.mean(loss_act_hist), 'timestep': it})
            wandb.log({'loss_crit': np.mean(loss_crit_hist), 'timestep': it})
            wandb.log({'alpha': alpha, 'timestep': it})

            loss_act_hist = []
            loss_crit_hist = []

            state = env.reset()
            total_reward = 0
            done = 0

        if it <= config['start_act']:
            action = env.action_space.sample()
        else:
            action = model.get_action(state, deterministic=False)

        state_p, reward, done, _ = env.step(action)

        # reward shaping
        reward += 300 * (config['gamma'] * np.abs(state_p[1]) - np.abs(state[1]))

        buffer.add((state, action, reward, state_p, done))
        state = state_p

        total_reward += reward
        if it > config['start_train']:
            batch = buffer.sample()
            loss_act, loss_crit = model.update(batch)

            loss_act_hist.append(loss_act)
            loss_crit_hist.append(loss_crit)

            if it % config['test_every'] == 0:
                reward_mean, reward_std = test(model, config['env'])
                reward_mean_hist.append(reward_mean)
                reward_std_hist.append(reward_std)
                timestep_hist.append(it)

                alpha = model.get_alpha()

                print(
                    f'Timestep: {it}\t Reward (mean): {reward_mean:.5f}\t\
                     Reward (std): {reward_std:.5f}\t Entropy Temp: {alpha:.5f}')
                wandb.log({'reward_test': reward_mean, 'timestep': it})

    env.close()
    model.save_model('.')

    return model, reward_mean_hist, reward_std_hist, timestep_hist


def test(model, env, epochs=5):
    reward_hist = []

    seed = hash(f'Please work SAC') % 1024 - 1

    env = gym.make(env)
    env.action_space.seed(seed)
    env.seed(seed)

    for e in range(epochs):
        state = env.reset()

        done = 0
        reward_total = 0

        while not done:
            action = model.get_action(state, deterministic=True)

            state, reward, done, _ = env.step(action)
            reward_total += reward

        reward_hist.append(reward_total)

    return np.mean(reward_hist), np.std(reward_hist)


# taken from https://stackoverflow.com/a/57452630
# class GumbelSoftmax(torch.distributions.RelaxedOneHotCategorical):
#     def sample(self, sample_shape=torch.Size()):
#         """Gumbel-softmax sampling. Note rsample is inherited from RelaxedOneHotCategorical"""
#         u = torch.empty(self.logits.size(), device=self.logits.device, dtype=self.logits.dtype).uniform_(0, 1)
#         noisy_logits = self.logits - torch.log(-torch.log(u))
#         return torch.argmax(noisy_logits, dim=-1)
#
#     def log_prob(self, value):
#         """value is one-hot or relaxed"""
#         if value.shape != self.logits.shape:
#             value = F.one_hot(value.long(), self.logits.shape[-1]).float()
#             assert value.shape == self.logits.shape
#         return - torch.sum(- value * F.log_softmax(self.logits, -1), -1)


def plot(x, reward_mean, reward_std):
    reward_mean, reward_std = np.array(reward_mean), np.array(reward_std)

    plt.figure(dpi=150)
    plt.plot(x, reward_mean, label='SAC', marker='o')
    plt.fill_between(x, reward_mean - 3 * reward_std, reward_mean + 3 * reward_std,
                     alpha=0.15)
    plt.tight_layout()
    plt.legend();

    locs, labels = plt.xticks()
    xlabels = [f'{x:.0f}k' for x in locs / 1000]
    plt.xticks(locs, xlabels);

    ax = plt.gca()
    ax.autoscale()
