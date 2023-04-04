import gym
from gym.wrappers import FlattenObservation
from src import *

# env generation:
env = gym.make('MountainCar-v0')
steps = 500_000

agent_left = PPO_Agent("MlpPolicy", # one of "MlpPolicy", "CnnPolicy"
                      env.observation_space,
                      env.action_space,
                      hook = Reward_Per_Episode_Hook()
                 )
agent_right = PPO_Agent("MlpPolicy", # one of "MlpPolicy", "CnnPolicy"
                      env.observation_space,
                      env.action_space,
                      hook = Reward_Per_Episode_Hook()
                 )
def determine_active(state): # only vlaid for mountain car
    if state[1] < 0:
        return 0
    else:
        return 1
def reward_left(s,a,ns):
    return -ns[1] # positive reward for going left
def reward_right(s,a,ns):
    return ns[1] # positive reward for going right
agent = Compositional_Agent([agent_left, agent_right],
                            determine_active,
                            [reward_left, reward_right],
                            hook=Reward_Per_Episode_Hook()
                            )

run(env, agent, steps=steps)

# see results
agent.plot()
run(env, agent, 1_000, render=True, train=False)
