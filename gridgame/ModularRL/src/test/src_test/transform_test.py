import gym
from gym.wrappers import FlattenObservation
from src import *
from gym.spaces.box import Box

# env generation:
env = gym.make('Acrobot-v1')

ppo_agent = PPO_Agent("MlpPolicy", # one of "MlpPolicy", "CnnPolicy"
                  Box(env.observation_space.low[:-1], env.observation_space.high[:-1]),
                  env.action_space, # Box(env.action_space.low * 2, env.action_space.high * 2),
                  hook = Reward_Per_Episode_Hook()
                 )
agent = State_Action_Transforming_Agent( ppo_agent,
                                         state_transformation=lambda x: x[:-1],
                                         # action_transformation_agent_to_env=lambda x: x / 2.0,
                                         # action_transformation_env_to_agent=lambda x: x * 2.0
                                         )


run(env, agent, steps=500_000)

# see results
agent.plot()
run(env, agent, 1_000, render=True, train=False)
