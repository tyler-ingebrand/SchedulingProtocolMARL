import enum

from gymnasium.spaces import MultiDiscrete, Discrete

from src.Agents.Abstract_Agent import Abstract_Agent
from src.Hooks.Abstract_Hook import Abstract_Hook
from src.Hooks.Do_Nothing_Hook import Do_Nothing_Hook
import torch

# Create a replay buffer to store experiences. This assumes state and actions are discrete (ints)
class Replay_Buffer:
    def __init__(self, state_dims, action_dims, buffer_size=100_000, device=None):
        self.buffer_size = buffer_size
        self.device = torch.device(device if device is not None else "cuda:0" if torch.cuda.is_available() else "cpu")

        state_dim_len = len(state_dims)
        action_dim_len = len(action_dims)

        # Create tables
        self.states = torch.zeros((self.buffer_size, state_dim_len), device=self.device, dtype=torch.long) # int type
        self.actions = torch.zeros((self.buffer_size, action_dim_len), device=self.device, dtype=torch.long) # int type
        self.rewards = torch.zeros((self.buffer_size,), device=self.device)
        self.dones = torch.zeros((self.buffer_size,), device=self.device)
        self.next_states = torch.zeros((self.buffer_size, state_dim_len), device=self.device, dtype=torch.long) # int type

        # create counter of where we are. Need to know if we are full or not for sampling
        self.pos = 0
        self.full = False

    # add data to our buffer
    # wrap around if needed
    def add(self, state, action, reward, done, next_state):
        self.states[self.pos] = torch.tensor(state, dtype=torch.long, device=self.device)
        self.actions[self.pos] = torch.tensor(action, dtype=torch.long, device=self.device)
        self.rewards[self.pos] = torch.tensor(reward, device=self.device)
        self.dones[self.pos] = torch.tensor(done, device=self.device)
        self.next_states[self.pos] = torch.tensor(next_state, dtype=torch.long, device=self.device) if next_state is not None else torch.zeros_like(self.states[self.pos])
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0


    # sample a random batch of data from the dataset
    def sample(self, batch_size):
        batch_inds = torch.randint(low=0,
                                   high=self.pos if not self.full else self.buffer_size, # exclusive
                                   size=(batch_size,))
        data = (self.states[batch_inds],
                self.actions[batch_inds],
                self.rewards[batch_inds],
                self.dones[batch_inds],
                self.next_states[batch_inds])
        return data






# Agent interface
# this class provides a tabular Q based algorithm for solving discrete state/action space MDPs
class Tabular_Q_Agent(Abstract_Agent):
    class ExplorationType(enum.Enum):
        Epsilon = 1
        Boltzmann = 2
        Deterministic = 3
    # for now assume state_space is multi-discrete and action space is discrete
    def __init__(self,
                 state_space,
                 action_space,
                 table_learning_rate=0.8,
                 gamma=0.9,
                 buffer_size=100_000,
                 update_frequency = 1000,
                 batch_size=100,

                 device=None,

                 reward_scale = 1.0, # for boltzman exploration
                 epsilon = 0.1, # for epsilon exploration
                 exploration_type = ExplorationType.Epsilon,

                 hook:Abstract_Hook = Do_Nothing_Hook()):

        self.hook = hook

        # detirmine state dims. Must be a tuple
        if type(state_space) is MultiDiscrete:
            self.state_dims = tuple(state_space.nvec)
        elif type(state_space) is Discrete:
            self.state_dims=(state_space.n,)
        else:
            raise Exception("Unknown state space type. Expected MultiDiscrete or Discrete, got {}".format(type(state_space)))

        # detirmine action dims. Must be a tuple
        if type(action_space) is MultiDiscrete:
            self.action_dims = tuple(action_space.nvec)
        elif type(action_space) is Discrete:
            self.action_dims = (action_space.n,)
        else:
            raise Exception("Unknown action space type. Expected MultiDiscrete or Discrete, got {}".format(type(action_space)))

        # learning hyper parameters
        self.lr = table_learning_rate
        self.gamma = gamma
        self.update_frequency = update_frequency
        self.current_step = 0
        self.batch_size = batch_size

        self.epsilon = epsilon
        self.reward_scale = reward_scale
        self.exploration_type = exploration_type

        # store memories in table
        self.device = torch.device(device if device is not None else "cuda:0" if torch.cuda.is_available() else "cpu")
        self.buffer = Replay_Buffer(self.state_dims, self.action_dims, buffer_size=buffer_size, device=device)

        # q table and optimizer. Can use gradient descent, ADAM, etc
        self.q_function = torch.zeros(self.state_dims + self.action_dims, device=self.device, requires_grad=True)
        self.q_optimizer = torch.optim.Adam((self.q_function,))

    # returns  an action for the given state.
    # Must also return extras, None is ok if the alg does not use them.
    
    def act(self, state):
        '''
        I think this is currently epsilon greedy exploration. probably could do better? 
        return: action, agent extras 
        '''
        with torch.no_grad():
            state = tuple(state)
            values = self.q_function[state]
            if self.exploration_type == self.ExplorationType.Boltzmann:
                values *= self.reward_scale
                probabilities = torch.nn.Softmax(dim=0)(values) # assumes 1 dimensional action space, todo
                random_index =  torch.multinomial(probabilities, num_samples=1)
                return random_index.item(), None
            elif self.exploration_type == self.ExplorationType.Epsilon and torch.rand(size=(1,)) <= self.epsilon:
                return torch.randint(0, self.action_dims[0], size=(1,)).item(), None
            else: # detirministic mode, or epsilon but not randomly selected
                return torch.max(values, dim=0).indices.item(), None



        
    # The main function to learn from data. At a high level, takes in a transition (SARS) and should update the function
    # ocassionally updates the policy, but always stores transition

    # TODO: check learn return arguments and update frequency also what are these inputs
    def learn(self, state, action, reward, next_state, done,truncated, info, extras, tag = "1"):
        
        ''' 
        What is the 'tag'
        what is extras 
        what is info
        what is done
        what is this "ocassionally updates the policy, but always stores transition"
        '''
        # allow hook to observe learning, mostly to save reward
        # note the agent needs to view the agent maybe, which is self
        self.hook.observe(self, state, action, reward, done, truncated,info, tag)

        # Save transition to buffer for future use
        self.buffer.add(state, action, reward, done, next_state)

        # increment step counter, every <update_frequency> steps, also update the table
        self.current_step += 1
        if self.current_step % self.update_frequency == 0:
            self.update_table()




    def update_table(self):
        # fetch data from table
        states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size)

        # delete old gradients. Necessary for torch syntax.
        self.q_optimizer.zero_grad()

        # compute q value according to reward + value at next state
        next_state_values = multi_dimensional_index_select(self.q_function, next_states)
        target_q_values = rewards + (1-dones) * self.gamma * torch.max(next_state_values, dim=1).values

        # current estimate for q value
        dims = torch.concat((states, actions), dim=1)
        current_q_values = multi_dimensional_index_select(self.q_function, dims)

        # loss is MSE of the difference
        loss = torch.nn.MSELoss()(current_q_values, target_q_values)

        # compute gradients and update table
        loss.backward()
        self.q_optimizer.step()


    def plot(self):
        print("plotting q")
        print(self.q_function)
        self.hook.plot()

def multi_dimensional_index_select(source, indices):
    output_dims_to_keep = len(source.shape) - indices.shape[1]
    lengths = source.shape[len(source.shape)-output_dims_to_keep:]
    ret = torch.empty((indices.shape[0],) + lengths, device=source.device)

    for batch in range(indices.shape[0]):
        out = source[tuple(indices[batch])]
        ret[batch] = out
    return ret

