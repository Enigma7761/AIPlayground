import copy
import torch
import numpy as np
from replay import SimpleReplay
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch import Tensor
from rl_algs.base_atari_model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DQN:
    def __init__(self, obs_space, action_space, model=None, replay_size=1000000, batch_size=32, optimizer=optim.RMSprop,
                 gamma=0.99, target_update_freq=10000, lr=2.5e-4, momentum=0.95, epsilon_max=1, epsilon_min=0.1,
                 epsilon_frames=1000000, update_freq=4, frame_skip=4, replay_start_size=50000):
        """
        Initializes a DQN agent, with some default parameters, as well as information about the action
        and observation space of the environment.
        :param obs_space: The observation space of the environment (tuple of dimensions)
        :param action_space: The action space (integer)
        :param model: The model structure (None for default model, or optionally pass a model)
        :param replay_size: The replay size (default 1 million)
        :param batch_size: The batch size (default 32)
        :param optimizer: The optimizer (default RMSProp)
        :param gamma: The reward discount factor (default 0.99)
        :param target_update_freq: The frequency at which the target model is updated at (default 10,000)
        :param lr: The learning rate (default 0.00025)
        :param epsilon_max: The maximum epsilon, also the intial epsilon value (default 1)
        :param epsilon_min: The minimum epsilon (default 0.1)
        :param epsilon_frames: The amount of frames to decrease epsilon over until epsilon min (default 1 million)
        :param update_freq: The frequency to update the model parameters at (default 4)
        :param frame_skip: The number of frames skipped (default 4)
        :param replay_start_size: The initial size of the replay (default 1 million)
        """
        if model is None:
            self.model = Model(frame_skip, action_space)
            self.target = copy.deepcopy(self.model)
        else:
            self.model = model
            self.target = model
        self.obs_space = obs_space
        self.action_space = action_space
        self.epsilon = epsilon_max
        self.replay = SimpleReplay(size=replay_size, batch_size=batch_size)
        self.optim = optimizer(self.model.parameters(), lr=lr, momentum=momentum)
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_frames = epsilon_frames
        self.target_update_freq = target_update_freq
        self.iter = 0
        self.update_freq = update_freq
        self.frame_skip = frame_skip
        self.epsilon_dec = (epsilon_max - epsilon_min) * frame_skip / epsilon_frames
        self.replay_start_size = replay_start_size

    def choose_action(self, obs):
        """
        Chooses an action for the current observation.
        :param obs: The observation
        :return: The chosen action with epsilon-random exploration
        """
        rand = np.random.random()
        if rand < self.epsilon:
            return np.random.randint(self.action_space, size=1)
        else:
            obs = torch.tensor(obs).to(device)
            return self.model(obs).argmax().detach().numpy()

    def step(self, obs, action, reward, next_obs, done):
        """
        Take a single step for the agent
        :param obs: The observation
        :param action: The action
        :param reward: The reward
        :param next_obs: The next observation
        :param done: If the environment is done
        """
        self.iter += 1
        self.replay.store(obs, action, reward, next_obs, done)
        self.decrease_epsilon()
        if self.iter % self.update_freq == 0 and len(self.replay) > self.replay_start_size:
            self.train()
            if self.iter // self.update_freq % self.target_update_freq:
                self.update_target()

    def train(self):
        """
        Trains the main model on a random sampling from the replay.
        """
        self.optim.zero_grad()
        obs, action, reward, next_obs, done = self.replay.sample()
        obs = torch.tensor(obs).to(device)
        action = torch.tensor(action).to(device)
        reward = torch.tensor(reward).to(device)
        next_obs = torch.tensor(next_obs).to(device)
        done = torch.tensor(done).to(device)

        target = reward + self.gamma * torch.max(self.target(next_obs), dim=0) * \
                 (done.type(torch.int) == 0).type(torch.float32)

        vals = torch.index_select(self.model(obs), 1, action)

        loss = F.mse_loss(target, vals)
        loss.backward()
        self.optim.step()

    def update_target(self):
        """
        Copies the current model onto the target model.
        """
        self.target = copy.deepcopy(self.model)

    def decrease_epsilon(self):
        """
        Linearly decreases epsilon until epsilon_min.
        """
        self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_min)

    def eval_choose_action(self, obs):
        """
        For evaluating the performance of the model with no exploration.
        :param obs: The observation of the state
        :return: The chosen action
        """
        obs = torch.tensor(obs).to(device)
        return self.model(obs).argmax().detach().numpy()

