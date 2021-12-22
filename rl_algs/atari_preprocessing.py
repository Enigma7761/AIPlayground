import gym
import numpy as np
from collections import deque


class AtariWrapper:
    def __init__(self, env):
        self.env = env
        self.obs = deque(maxlen=4)

    def step(self, action):
        
    def reset(self):
