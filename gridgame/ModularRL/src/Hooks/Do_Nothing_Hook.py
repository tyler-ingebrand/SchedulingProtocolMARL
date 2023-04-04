from src.Hooks.Abstract_Hook import Abstract_Hook
# from ModularRL import Abstract_Hook

# A hook which does nothing. Used as a default when we dont want a hook
# But we want the variable to not be None to avoid an error.
class Do_Nothing_Hook(Abstract_Hook):
    # does nothing
    def __init__(self):
        pass

    # Does nothing
    def observe(self, agent, obs, action, reward, done,truncated, info, tag = "1"):
        pass

    # returns nothing
    def get_output(self):
        return None

    # plots whatever data is collected
    def plot(self):
        # print(" I tried to plot a do nothing")
        pass