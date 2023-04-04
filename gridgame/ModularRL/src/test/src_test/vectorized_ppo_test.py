import gym
from gym.wrappers import FlattenObservation
from src import *
from stable_baselines3.common.env_util import make_vec_env

# This cannot work becuase PPO exploits the ordering of data in the buffer
# Since this adds N env data to the buffer at the same time, it is impossible for this ordering to be exploited
# We would need to do some structure to use a buffer for each env. However, when we get to compositionality, this would not be possible
# Since a componenent may only be active in some of the envs
# Maybe store the entire episode, and then add it onces its finished.
# otherwise I have no ideas.


env = make_vec_env("Pendulum-v1", n_envs=4, seed=0)
steps = 100_000

inner_agent = PPO_Agent("MlpPolicy", # one of "MlpPolicy", "CnnPolicy"
                  env.observation_space,
                  env.action_space,
                  hook = Reward_Per_Episode_Hook()
                 )
agent = Vectorized_Agent(inner_agent, 4, env.action_space)

run(env, agent, steps=steps)

# see results
agent.plot()
run(gym.make("Pendulum-v1"), agent.agent, 1_000, render=True, train=False)
