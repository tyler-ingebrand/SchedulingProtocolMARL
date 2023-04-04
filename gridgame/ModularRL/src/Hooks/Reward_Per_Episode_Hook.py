import matplotlib.pyplot as plt
from src.Hooks.Abstract_Hook import Abstract_Hook
import numpy

# A hook which keeps track of reward per episode in training
# This is in training mode only, no evaluation is performed. Also this only works for 1 env at a time
# Also keeps track of number of steps and episodes observed
# Outputs a dict containing "episode_rewards", "n_steps", "n_episodes"

class Reward_Per_Episode_Hook(Abstract_Hook):
    def __init__(self):
        self.current_episode_reward = {}
        self.rewards = []
        self.number_steps = 0
        self.number_episodes = 0

    def observe(self, agent, obs, action, reward, done,truncated, info, tag = "1"):
        self.current_episode_reward[tag] = self.current_episode_reward.get(tag, 0) + reward
        self.number_steps += 1
        if done or truncated:
            self.number_episodes += 1
            self.rewards.append(self.current_episode_reward.get(tag, 0))
            self.current_episode_reward[tag] = 0

    def get_output(self):
        return {"episode_rewards" : self.rewards,
                "n_steps" :         self.number_steps,
                "n_episodes":       self.number_episodes}

    def plot(self):
        plt.plot(self.rewards)
        plt.show()
