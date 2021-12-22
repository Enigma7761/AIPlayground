import gym
from rl_algs import DQN
import numpy as np
from matplotlib import pyplot as plt
"""
Simple experiment running the default parameters of the DQN and the default model (detailed in the original paper 
https://www.nature.com/articles/nature14236) on Breakout
"""

env = gym.make("BreakoutDeterministic-v0")


agent = DQN(env.observation_space.shape, env.action_space.n)
eval_reward = []
epochs = 100

"""
Training for "epochs" games and evaluating every 100 games
"""
for i in range(1, epochs):
    obs = env.reset()
    done = [False]
    while done[0] is False:
        action = agent.choose_action(obs)
        next_obs, reward, done, info = env.step(action)
        agent.step(obs, action, reward, next_obs, done)

        obs = next_obs

    if i % 100 == 0:
        for _ in range(10):
            obs = env.reset()
            done = False
            total_reward = 0
            while done[0] is False:
                action = agent.eval_choose_action(obs)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
            eval_reward.append(total_reward / 10)

plt.plot(range(100, epochs, 100), eval_reward)
plt.xlabel("Epoch")
plt.ylabel("Average Reward")
plt.show()
