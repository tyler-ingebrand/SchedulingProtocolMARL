from src.Hooks.Abstract_Hook import Abstract_Hook

# A hook which simply wraps multiple other hooks at once.
# Calls all hooks whenever this hook is called.
class Multi_Hook(Abstract_Hook):
    # Save the list of hooks
    def __init__(self, list_o_hooks):
        self.hooks = list_o_hooks

    # Send observations to all hooks
    def observe(self,  agent, obs, action, reward, done,truncated, info, tag = "1"):
        for h in self.hooks:
            h.observe(agent,obs,action,reward,done,truncated,info, tag)

    # Get output from a hooks, return them as a list
    def get_output(self):
        return (h.get_output() for h in self.hooks)

    # plots each hook
    def plot(self):
        for h in self.hooks:
            h.plot()

