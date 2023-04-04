from src.Hooks.Abstract_Hook import Abstract_Hook
from src.Hooks.Do_Nothing_Hook import Do_Nothing_Hook

# Agent interface
class Abstract_Agent:

    # returns  an action for the given state.
    # Must also return extras, None is ok if the alg does not use them.
    def act(self, state):
        raise Exception("Unimplemented")

    # The main function to learn from data. At a high level, takes in a transition (SARS) and should update the function
    # ocassionally updates the policy, but always stores transition
    def learn(self, state, action, reward, next_state, done, truncated, info, extras, tag = "1"):
        raise Exception("Unimplemented")

    def plot(self):
        raise Exception("Unimplemented")
