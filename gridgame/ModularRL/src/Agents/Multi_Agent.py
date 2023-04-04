from typing import Dict

from src.Agents.Abstract_Agent import Abstract_Agent
from src.Hooks.Abstract_Hook import Abstract_Hook
from src.Hooks.Do_Nothing_Hook import Do_Nothing_Hook

# Agent interface
class Multi_Agent(Abstract_Agent):

    def __init__(self, agents:Dict,
                 hook : Abstract_Hook = Do_Nothing_Hook() # The hook to observe this process
                 ):
        self.hook = hook
        self.agents = agents


    # returns  an action for the given state.
    # Must also return extras, None is ok if the alg does not use them.
    
    def act(self, state):
        actions = {}
        extras = {}
        for key in state:
            agent_action, agent_extras = self.agents[key].act(state[key])
            actions[key] = agent_action
            extras[key] = agent_extras
        return actions, extras

    # The main function to learn from data. At a high level, takes in a transition (SARS) and should update the function
    # ocassionally updates the policy, but always stores transition


    def learn(self, state, action, reward, next_state, done, truncated, info, extras, tag = "1"):
        # update hook
        # print("reward is", reward)
        self.hook.observe(self, state, action, reward, done, truncated, info, tag)

        # Update all agents
        for agent in state:
            self.agents[agent].learn(state.get(agent), action.get(agent), reward.get(agent), next_state.get(agent), done.get(agent),truncated.get(agent), info.get(agent), extras.get(agent))

    def plot(self):
        self.hook.plot()
        for a in self.agents:
            self.agents[a].plot()

