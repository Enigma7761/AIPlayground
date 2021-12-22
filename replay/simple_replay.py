import numpy as np


class SimpleReplay:
    """
    A simple replay buffer for the base DQN. Stores the observations, actions, and other information
    about the environment and the agent.
    """
    def __init__(self, size=1000000, batch_size=32):
        """
        Initializes the replay buffer to have a size and the batch size for later sampling.
        :param size: Size of the buffer
        :param batch_size: The batch size when sampling
        """
        self.size = size
        self.obs = np.empty(size, dtype=object)
        self.actions = np.empty(size, dtype=object)
        self.rewards = np.empty(size, dtype=object)
        self.next_obs = np.empty(size, dtype=object)
        self.dones = np.empty(size, dtype=bool)
        self.batch_size = batch_size
        self.index = 0
        self.full = False

    def store(self, obs, action, reward, next_obs, done):
        """
        Stores the values into the replay buffer.
        :param obs: The current observation
        :param action: The action of the agent
        :param reward: The reward from the environment
        :param next_obs: The next observation
        :param done: If the environment is "done"
        """
        self.obs[self.index] = obs
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_obs[self.index] = next_obs
        self.dones[self.index] = done
        self.index = (self.index + 1) % self.size
        if self.index == 0:
            self.full = True

    def __len__(self):
        """
        Returns the length of the buffer.
        :return: The length of the buffer
        """
        return self.obs.shape[0] if self.full else self.index

    def sample(self):
        """
        Returns a random sample of size batch_size from the buffer.
        :return: The random sample of collections from the buffer
        """
        indices = np.random.randint(low=0, high=len(self), size=self.batch_size, dtype=int)

        obs = self.obs[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_obs = self.next_obs[indices]
        dones = self.dones[indices]

        return obs, actions, rewards, next_obs, dones


if __name__ == "__main__":
    """
    Only to test the speed of the replay buffer and sampling.
    """
    import time

    replay = SimpleReplay()
    for i in range(replay.size):
        replay.store(np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn())
    print("Starting timing")
    start_time = time.time()
    for i in range(1000):
        replay.sample()
    print(f'{time.time() - start_time} seconds')
