import gym
import numpy as np
from collections import deque
import cv2

"""
Does the basic atari preprocessing, accounting for the different kinds of atari environments you can feed into it.
Source:
https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
"""


class AtariWrapper:
    def __init__(self, env, frameskip=4, stack=4, noop_max=30):
        """
        (From https://github.com/openai/gym/issues/1280)
        v0 vs v4: v0 has a probability to repeat the action (ignore the action and use previous action)
        Deterministic vs NoFrameSkip vs base:
            Deterministic (i.e. 'BreakoutDeterministic') has a fixed frameskip of 4
            NoFrameSkip doesn't do frameskipping
            base (i.e. 'Breakout') does frameskipping uniformly sampled from [2, 4]
        :param env: A string for which atari environment the environment will be of (i.e. 'Breakout-v0')
        :param frameskip: How many frames will be ignored in the observation. Also the number of times an action will
        be repeated. (only matters if the environment doesn't do frameskipping)
        :param stack: The number of frames stacked together in an observation returned from the wrapper
        :param noop_max: The number of "no-operations" after reset.
        """
        self.env = gym.make(env)
        self.ale = env.unwrapped.ale
        self.obs = np.empty((stack, 84, 84), dtype=np.uint8)
        self.frameskip = frameskip
        self.noop_max = noop_max
        if 'NoFrameSkip' in env:
            self.mode = 0
        elif 'Deterministic' in env:
            self.mode = 1
        else:
            self.mode = 2

    def step(self, action):
        """
        Replaces the old step function for the environment, functioning in the same way but
        returning the preprocessed data.
        :param action: The action to perform in the environment
        (potentially multiple times for frameskipping or repeated actions)
        :return: The observation, reward, if the environment is 'done', and info about the environment, after the action
        is performed.
        """
        if self.mode == 0 or self.mode == 2:
            _, reward, done, info = self.env.step(action)
            obs = self.ale.getScreenGrayscale()
            obs = self.shrink(obs)
        else:
            total_reward = 0
            for _ in range(self.frameskip - 2):
                _, reward, done, info = self.env.step(action)
                total_reward += reward

            _, reward, done, info = self.env.step(action)
            obs1 = self.ale.getScreenGrayscale()
            obs1 = self.shrink(obs1)
            total_reward += reward
            _, reward, done, info = self.env.step(action)
            obs2 = self.ale.getScreenGrayscale()
            obs2 = self.shrink(obs2)
            total_reward += reward
            obs = self.component_wise_max(obs1, obs2)
        self.obs[:-1] = self.obs[1:]
        self.obs[-1] = obs
        return self.obs, total_reward, done, info

    def reset(self):
        """
        Resets the environment and performs a random number of no-op actions in the environment, either enough to
        fill self.obs or <= noop_max actions in the environment.
        :return: The initial state after reset.
        """
        self.env.reset()
        noop = np.random.randint(0, self.noop_max)
        if self.mode == 0 or self.mode == 1:
            for i in range(0, max(self.obs.shape[0], noop // 4)):
                obs, _, _, _ = self.step(0)
        else:
            for i in range(0, max(self.obs.shape[0], noop // 3)):
                obs, _, _, _ = self.step(0)
        return self.obs

    def shrink(self, obs):
        """
        Shrinks the image to an 84 by 84 image. cv2 removes the channel so we add it back in dim 0.
        :param obs: The observation from the environment
        :return: The shrunken image.
        """
        return cv2.resize(obs, (84, 84))[np.newaxis, :, :]

    def component_wise_max(self, obs1, obs2):
        """
        Does component-wise maximum for the images between the last 2 frames. Only happens if the environment
        doesn't automatically do frameskipping.
        :param obs1: The first observation
        :param obs2: The second observation
        :return: The observation after the component-wise maximum.
        """
        obs = np.append(obs1, obs2, axis=0)
        obs = np.max(obs, axis=0)[np.newaxis, :, :]
        return obs
