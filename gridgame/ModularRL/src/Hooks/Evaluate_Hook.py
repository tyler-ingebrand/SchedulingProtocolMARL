import matplotlib.pyplot as plt
from src.Hooks.Abstract_Hook import Abstract_Hook
import numpy

# A hook which keeps track of reward per episode in training
# This is in training mode only, no evaluation is performed. Also this only works for 1 env at a time
# Also keeps track of number of steps and episodes observed
# Outputs a dict containing "episode_rewards", "n_steps", "n_episodes"
class Evaluate_Hook(Abstract_Hook):
    def __init__(self, eval_env, every_N_steps=10000, eval_runs=10):
        self.rewards = []
        self.number_steps = 0
        self.every_N_steps = every_N_steps
        self.eval_env = eval_env
        self.eval_runs = eval_runs
    def observe(self, agent, obs, action, reward, done,truncated, info, tag = "1"):
        self.number_steps += 1
        if self.number_steps % self.every_N_steps == 0:
            self.evaluate(agent)

    def get_output(self):
        return {"evaluation_episode_rewards" : self.rewards}

    def plot(self):
        plt.plot(self.rewards)
        plt.show()

    def evaluate(self, agent):
        episode_rewards = []

        # run some number of episodes
        for i in range(self.eval_runs):
            rewards = 0
            done = False
            obs = self.eval_env.reset()
            # run the episode
            while not done:
                action, value, log_action_probs = agent.act(obs)
                nobs, reward, done, info = self.eval_env.step(action)
                rewards += reward
                obs = nobs
            # remember the total reward
            episode_rewards.append(rewards)

        self.rewards.append(sum(episode_rewards)/self.eval_runs) # append average reward to our list