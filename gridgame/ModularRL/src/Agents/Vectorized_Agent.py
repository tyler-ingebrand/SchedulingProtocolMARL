from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor, configure_logger
from .Abstract_Agent import Abstract_Agent
from src.Hooks.Abstract_Hook import *
from src.Hooks.Do_Nothing_Hook import *
import numpy

class Vectorized_Agent(Abstract_Agent):
    def __init__(self,
                 agent,
                 number_envs, # how many environments we are handling
                 action_space,
                 hook : Abstract_Hook = Do_Nothing_Hook() # The hook to observe this process
                 ):
        raise Exception("This does not work at the moment with PPO")
        self.hook = hook
        self.agent = agent
        self.number_envs = number_envs
        self.action_space = action_space

    def act(self, state):

        # actions = numpy.zeros((self.number_envs, self.action_space.shape[0]))
        actions = []
        extras = []
        for env_index in range(self.number_envs):
            action, extra = self.agent.act(state[env_index])
            # actions[env_index] = action
            actions.append(action)
            extras.append(extra)
        return actions, extras

    def learn(self, state, action, reward, next_state, done, truncated,info, extras, tag = "1"):
        # Allow hook to record learning
        self.hook.observe(self, state, action, reward, done,truncated, info, tag)

        for env_index in range(self.number_envs):
            self.agent.learn(state[env_index],
                             action[env_index],
                             reward[env_index],
                             next_state[env_index],
                             done[env_index],
                             info[env_index],
                             extras[env_index],
                             str(env_index)) # append the env number as a string
                            # this allows us hook for the agent to keep track of the different env indices

    def plot(self):
        self.hook.plot()
