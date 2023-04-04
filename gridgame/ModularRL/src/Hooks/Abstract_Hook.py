# A hook for the training loop. Gets access to all of the variables,
# can do some processing such as recording reward, testing, etc. Should not change the env or agent.
# This provides an interface for creating a hook.
class Abstract_Hook:
    # Update the hook with an observation of what is happening in the training cycle
    def observe(self, agent, obs, action, reward, done, truncated,info, tag = "1"):
        raise Exception("Unimplemented")

    # Return the result of the hooks observations. Can be anything, depending on what the hook is for
    def get_output(self):
        raise Exception("Unimplemented")

    # plots whatever data is collected
    def plot(self):
        raise Exception("Unimplemented")
