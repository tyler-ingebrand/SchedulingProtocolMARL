from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor, configure_logger
from .Abstract_Agent import Abstract_Agent
from src.Hooks.Abstract_Hook import *
from src.Hooks.Do_Nothing_Hook import *
import gym
from gym.spaces import Discrete
import numpy
import torch
# Stable baselines 3 expects a gym env for input. But it only uses the state and action space
# so we can generate a dummy env from a state and action space to use in the constructor
class DummyEnv(gym.Env):
    def __init__(self, state_space, action_space):
        self.observation_space = state_space
        self.action_space = action_space

class PPO_Agent(Abstract_Agent):
    def __init__(self, policy_type, state_space, action_space, hook : Abstract_Hook = Do_Nothing_Hook()):
        dummy_env = DummyEnv(state_space, action_space)
        self.alg = PPO(policy_type, dummy_env)
        self.state_space = state_space
        self.action_space = action_space
        self.hook = hook
        self.is_discrete_action = type(action_space) is Discrete
        self.last_done = True
        # Produce logger because this is not done in consturctor for some reason
        if not self.alg._custom_logger:
            self.alg._logger = configure_logger(self.alg.verbose, self.alg.tensorboard_log, "Test", 10000)

    def act(self, state):
        assert state in self.state_space, "State is {}, expected something in {}".format(state, self.state_space)
        state_on_device = obs_as_tensor(state.reshape(1, self.state_space.shape[0]), self.alg.device)
        action, value, log_action_probs = self.alg.policy(state_on_device)
        action = action.detach().cpu().numpy().flatten()
        extras = {"value" : value.detach(),
                  "log_prob" : log_action_probs.detach()}

        # if discrete, convert to an int because env expects it
        if self.is_discrete_action:
            action = action[0]
        # If continuous, clip to range
        else:
            action = numpy.clip(action, self.action_space.low, self.action_space.high)
        return action, extras

    def learn(self, state, action, reward, next_state, done,truncated, info, extras, tag = "1"):
        assert state in self.state_space, "State is {}, expected something in {}".format(state, self.state_space)
        assert action in self.action_space, "Action is {}, expected something in {}".format(action, self.action_space)
        assert next_state in self.state_space, "Next_State is {}, expected something in {}".format(next_state, self.state_space)
        value, action_probabilities = extras["value"], extras["log_prob"]

        # Allow hook to observe
        self.hook.observe(self, state, action, reward, done,truncated, info, tag)

        # if discrete env, send back to tensor. Converted to a discrete int for the env in the act function.
        if self.is_discrete_action:
            action = action.reshape(-1, 1)

        # append value to reward if the next state is terminal
        if done: # currently assumes only ends due to timeout
            terminal_obs = self.alg.policy.obs_to_tensor(next_state)[0]
            with torch.no_grad():
                terminal_value = self.alg.policy.predict_values(terminal_obs)[0].cpu().numpy()
            reward += self.alg.gamma * terminal_value


        # add to rollout buffer
        # self.alg.rollout_buffer.add(state.transpose(), action, reward, False,           value, action_probabilities) # works on cont
        self.alg.rollout_buffer.add(state.transpose(), action, reward, self.last_done,  value, action_probabilities) # does not work on pendulum?
        self.last_done = done
        # if it has been some number of steps, train policy.
        if self.alg.rollout_buffer.full:
            self.alg.rollout_buffer.compute_returns_and_advantage(last_values=value, dones=done)
            self.alg.train()
            self.alg.rollout_buffer.reset()

    def plot(self):
        self.hook.plot()