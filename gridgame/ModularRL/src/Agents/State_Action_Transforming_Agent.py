from src.Agents.Abstract_Agent import Abstract_Agent
from src.Hooks.Abstract_Hook import Abstract_Hook
from src.Hooks.Do_Nothing_Hook import Do_Nothing_Hook

# Define identity function as default
def identity(x):
    return x

# Agent interface
class State_Action_Transforming_Agent(Abstract_Agent):

    def __init__(self,
                 agent : Abstract_Agent,
                 state_transformation = identity,
                 action_transformation_agent_to_env = identity,
                 action_transformation_env_to_agent = identity, # these should be inverses
                 hook: Abstract_Hook = Do_Nothing_Hook()
                 ):
        self.agent = agent
        self.state_transformation = state_transformation
        self.action_transformation_agent_to_env = action_transformation_agent_to_env
        self.action_transformation_env_to_agent = action_transformation_env_to_agent
        self.hook = hook

    # returns  an action for the given state.
    # Must also return extras, None is ok if the alg does not use them.

    def act(self, state):
        modified_state = self.state_transformation(state)
        agent_action, extras = self.agent.act(modified_state)
        env_action = self.action_transformation_agent_to_env(agent_action)
        return env_action, extras

    # The main function to learn from data. At a high level, takes in a transition (SARS) and should update the function
    # ocassionally updates the policy, but always stores transition
    def learn(self, state, action, reward, next_state, done, truncated, info, extras, tag = "1"):
        self.hook.observe(self, state, action, reward, done,truncated, info)
        modified_state = self.state_transformation(state)
        modified_next_state = self.state_transformation(next_state)
        modified_action = self.action_transformation_env_to_agent(action)
        self.agent.learn(modified_state, modified_action, reward, modified_next_state, done,truncated, info, extras)

    def plot(self):
        self.hook.plot()
        self.agent.plot()