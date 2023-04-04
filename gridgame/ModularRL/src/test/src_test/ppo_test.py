import gym
from gym.wrappers import FlattenObservation
from src import *

cont = False

# env generation:
# Use FlattenObservation on dict env, such as 'FetchPickAndPlace-v1': env = FlattenObservation(gym.make('FetchPickAndPlace-v1'))
if not cont:
    #env = gym.make("Acrobot-v1")
    env = gym.make('CartPole-v1')
    steps = 100_000
else:
    env = gym.make('Pendulum-v1') # does not work ATM
    steps = 500_000

agent = PPO_Agent("MlpPolicy", # one of "MlpPolicy", "CnnPolicy"
                  env.observation_space,
                  env.action_space,
                  hook = Reward_Per_Episode_Hook()
                 )
run(env, agent, steps=steps)

# see results
agent.plot()
run(env, agent, 1_000, render=True, train=False)
