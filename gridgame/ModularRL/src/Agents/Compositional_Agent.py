from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor, configure_logger
from .Abstract_Agent import Abstract_Agent
from src.Hooks.Abstract_Hook import *
from src.Hooks.Do_Nothing_Hook import *

 
class Compositional_Agent(Abstract_Agent):
    def __init__(self,
                 agents, # a list of agents to activate
                 determine_active_agent,  # A function of the state which returns the index of the active agent
                 reward_functions, # A list of reward functions to use, with indices matching the index of the agent
                 done_on_agent_transition, # whether or not to tell the agent it is done when the next active agent changes
                                            # should be true if transitioning is the goal, false if transitioning happens randomly
                 hook : Abstract_Hook = Do_Nothing_Hook(), # The hook to observe this process
                 update_memory=lambda state, memory : None,
                 ):
        self.hook = hook
        self.agents = agents
        self.determine_active_agent = determine_active_agent
        self.reward_functions = reward_functions
        self.update_memory = update_memory
        self.done_on_agent_transition = done_on_agent_transition
        self.memory = {}

    def act(self, state):
        self.memory = self.update_memory(state, self.memory)
        active_agent_index = self.determine_active_agent(state, self.memory)
        return self.agents[active_agent_index].act(state)

    def learn(self, state, action, reward, next_state, done, truncated, info, extras, tag = "1"):
        # Allow hook to record learning
        self.hook.observe(self, state, action, reward, done, truncated, info, tag)

        # Figure out which agent made the action, and which will make it at the next step
        current_active_agent = self.determine_active_agent(state, self.memory)
        next_active_agent = self.determine_active_agent(next_state, self.memory) 

        # done if MDP terminates or if the agent to act in next state is different, which means the sub-MDP has terminated
        current_done = done or (self.done_on_agent_transition and current_active_agent != next_active_agent)
        
        if next_active_agent == None:
            print(f" NEXT active agent State:{state}, next state {next_state}, done is {done}. ")
            print(f"current done is {current_done}")
        if current_active_agent == None:
            print(f" CURRENT active agent State:{state}, next state {next_state}, done is {done}. ")
            print(f"current done is {current_done}")
        # sometimes other information is needed that is not part of the state, action, next_state, such as goal information
        # which the agent is meant to learn implicitly, but is not provided in s,a,ns alone.
        reward = self.reward_functions[current_active_agent](state, action, next_state, info) #TODO: do not change reward. get rid of this line. put functions into env. 
        

        self.agents[current_active_agent].learn(state, action, reward, next_state, current_done, truncated, info, extras, tag)

    def plot(self):
        self.hook.plot()
        for a in self.agents:
            a.plot()