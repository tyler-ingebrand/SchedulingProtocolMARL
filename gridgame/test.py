import random

import gym.wrappers.monitoring.video_recorder
import gymnasium.spaces
import numpy as np
from env.video_recorder import RecordVideo
import pickle
from ModularRL.src.Agents.Multi_Agent import Multi_Agent
from ModularRL.src.Agents.Tabular_Q_Agent import Tabular_Q_Agent
from ModularRL.src.Agents.Compositional_Agent import Compositional_Agent
from ModularRL.src.Agents.State_Action_Transforming_Agent import State_Action_Transforming_Agent
from ModularRL.src.Core.run import run
from env.gridgame import GridGameEnvironment
from ModularRL.src.Hooks.Abstract_Hook import Abstract_Hook
from ModularRL.src.Hooks.Do_Nothing_Hook import Do_Nothing_Hook
from ModularRL.src.Hooks.Reward_Per_Episode_Hook import Reward_Per_Episode_Hook
from ModularRL.src.Hooks.Successful_Agents_Per_Episode_Hook import Successful_Agents_Per_Episode_Hook
import torch

'''
This file can run tests of agents with and without protocols. 
'''
def run_experiment(protocol,
                   seed,
                   number_steps=200_000,
                   device=None, # or "cuda:0", cuda:1, etc
                   game_size=10,
                   number_agents=15,
                   map_type='hallway',
                   save_video=False,
                   render=False,
                   ):
    assert protocol == "CSMA_CD" or protocol == "RR" or protocol == "STR" or protocol == "NoneBlind" or protocol == "NoneSeeing"
    print(f"Running {protocol}, seed {seed} for {number_steps} steps")
    number_testing_steps = 1000 # (for rendering)
    turn_length = 2 * int(game_size /3)
    timeout = 200

    # seed torch and np
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # set up environment
    env = GridGameEnvironment(number_agents = number_agents, size = game_size, timeout = timeout, render_mode = None, protocol = protocol, turn_length = turn_length, map_type = map_type)
    action_space = env.action_space(None)
    state_space = env.observation_space(None)
    tabular_q_state_space = gymnasium.spaces.MultiDiscrete(nvec=state_space.nvec[:-1]) # removes last dimension




    # Utils  CSMA-CD
    def update_memory_CSMA_CD(state, memory):
        memory["state"] = state

        # detect if we are in hallway AND someone else is:
        hallway_occupied = state[-1] == 1
        me_in_hallway = env.grid_size[0] / 3 <= state[0] < 2 * env.grid_size[0] / 3
        hallway_collision =  hallway_occupied and me_in_hallway

        # if collision, then I need to wait a random amount of time before trying to go through hallway
        # if not, then I can decrease my wait timer if I am waiting
        if hallway_collision:
            memory["wait_timer"] = memory.get("wait_timer", 0) + np.random.randint(1,5) #TODO: update wait time

        else:
            memory["wait_timer"] = max(memory.get("wait_timer", 0) - 1, 0) # cannot go below 0

        return memory

    def determine_active_CSMA_CD(state, memory):
        if state is None:
            return int(memory["wait_timer"]-1 > 0)

        # for next state. If wait timer will still be greater than 0, or hallway occupied, return 1. Else return 0
        if memory["state"] != state:
            return int(memory["wait_timer"]-1 > 0 or state[-1] == 1)

        # for current state. Simply check if wait_timer > 0 or someone currently in hallway
        return int(memory["wait_timer"] > 0 or state[-1] == 1)

    # Utils  RR
    def update_memory_RR(state, memory):
        '''
        Don't need any memory, RR turns are tracked by the environment.
        '''
        return None

    def determine_active_RR(state, memory):
        '''
        Tells Compositoinal agent who should get to be acting and who should be waiting.
        '''
        # will be 0 if it is the agent's turn, 1 if it is not the agent's turn.
        if state == None:
            return 1
        else:
            return state[-1]

    # Utils  STR
    def update_memory_STR(state, memory):
        '''
        Don't need any memory, RR turns are tracked by the environment.
        '''
        return None

    def determine_active_STR(state, memory):
        '''
        Tells Compositoinal agent who should get to be acting and who should be waiting.
        '''
        # will be 0 if it is the agent's turn, 1 if it is not the agent's turn.
        if state == None:
            return 1
        else:
            return state[-1]

    # Reward Utils
    def reward_empty_hallway(s, a, ns, info):
        if ns is None or info is None or (ns[0] == info[0] and ns[1] == info[1]):
            return 1 # reached the goal
        else:
            return -0.01

    def reward_occupied_hallway(s, a, ns, info):
        if ns is None or info is None or (ns[0] == info[0] and ns[1] == info[1]):
            return 1
        if (env.grid_size[0]/3)-1 <= ns[0] < (2 * env.grid_size[0]/3)+1:
            return -1
        else:
            return -0.01

    def reward_empty_hallway_RR(s,a, ns, info):
        if ns is None or info is None or (ns[0] == info[0] and ns[1] == info[1]):
            return 1 # reached the goal
        # TODO: I actually want to be penalized slightly for staying on my starting side when it is my turn to pass.
        # How do I get a record of my original side?
        starting_rooms = env.get_starting_side() # [1,2,1,2,1...] but I need to know which agent I am dealing with. Frick


    # function to Save Object (specifically hooks)
    def save_object(obj, filename):
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


    # Create agent holding all of the sub-agents, which hold 2 Q tables each
    # build agents
    if protocol != "NoneBlind" and protocol != "NoneSeeing":
        if protocol ==  'CSMA_CD':
            update_memory = update_memory_CSMA_CD
            determine_active = determine_active_CSMA_CD
        elif protocol == 'RR':
            update_memory = update_memory_RR
            determine_active = determine_active_RR
        elif protocol == 'STR':
            update_memory = update_memory_STR
            determine_active = determine_active_STR

        agent = Multi_Agent({
                        agent_name: Compositional_Agent(
                            agents=[State_Action_Transforming_Agent(agent=Tabular_Q_Agent(tabular_q_state_space, action_space, hook=Do_Nothing_Hook(),
                                                                                        exploration_type=Tabular_Q_Agent.ExplorationType.Epsilon, device=device),
                                                                    state_transformation=lambda state: state[:-1] if state is not None else None),
                                    State_Action_Transforming_Agent(agent=Tabular_Q_Agent(tabular_q_state_space, action_space, hook=Do_Nothing_Hook(),
                                                                                        exploration_type=Tabular_Q_Agent.ExplorationType.Epsilon, device=device),
                                                                    state_transformation=lambda state: state[:-1] if state is not None else None)],
                            determine_active_agent=determine_active,
                            reward_functions=[reward_empty_hallway,reward_empty_hallway],
                            update_memory = update_memory,
                            done_on_agent_transition=False # since transitions happen randomly, dont want to tell agent they reached a terminal state
                        )
                        for agent_name in env.possible_agents}, hook = Successful_Agents_Per_Episode_Hook()
        )



    elif protocol == "NoneBlind":
        agent = Multi_Agent({
            agent_name: Compositional_Agent(
                agents=[State_Action_Transforming_Agent(agent=Tabular_Q_Agent(tabular_q_state_space, action_space, hook=Do_Nothing_Hook(),
                                                                            exploration_type=Tabular_Q_Agent.ExplorationType.Epsilon, device=device),
                                                        state_transformation=lambda state: state[:-1] if state is not None else None),
                        ],
                determine_active_agent=lambda state, memory: 0,
                reward_functions=[reward_empty_hallway, ],
                update_memory=lambda state, memory: None,
                done_on_agent_transition=False
                # since transitions happen randomly, dont want to tell agent they reached a terminal state
            )
            for agent_name in env.possible_agents}, hook=Successful_Agents_Per_Episode_Hook()
        )
    elif protocol == "NoneSeeing":
        agent = Multi_Agent({
            agent_name: Compositional_Agent(
                agents=[Tabular_Q_Agent(state_space, action_space, hook=Do_Nothing_Hook(),
                                        exploration_type=Tabular_Q_Agent.ExplorationType.Epsilon, device=device),
                        ],
                determine_active_agent=lambda state, memory: 0,
                reward_functions=[reward_empty_hallway, ],
                update_memory=lambda state, memory: None,
                done_on_agent_transition=False
            )
            for agent_name in env.possible_agents}, hook=Successful_Agents_Per_Episode_Hook()
        )
    else:
        raise Exception(f"Invalid protocol: {protocol}")

    # set file names
    file_name = f"results/{number_agents}_agents_{protocol}_seed_{seed}.pkl"
    video_file_name_prefix = f"{number_agents}_agents_{protocol}_seed_{seed}"


    # run experiment
    run(env, agent, number_steps, train = True, show_progress=True)
    if render:
        agent.plot()

    # save multiagent hook
    save_object(agent.hook, file_name)
    print(f"Successfully saved multiagent hook at {file_name}")


    # render
    if save_video:
        render_env = GridGameEnvironment(number_agents = number_agents, size = game_size, render_mode="rgb_array")
        render_env = RecordVideo(render_env, video_folder="videos", name_prefix= video_file_name_prefix, episode_trigger=lambda step: True)
        run(render_env, agent, number_testing_steps, train = False, show_progress=True, verbose=True)

    if render:
        render_env = GridGameEnvironment(number_agents = number_agents, size = game_size, render_mode="human")
        run(render_env, agent, number_testing_steps, train = False, show_progress=True, verbose=True)

if __name__ == "__main__":
    run_experiment(protocol="RR", seed=0, number_steps=10_000, device="cpu", render=True)


# TODO: 
# [ ] need to move reward into the environment
#     [] split reward into two parts according to what the protocol says
#     []figure out what Compositional Agent is doing. 
# [x] make type of environment according to parameter. OR make new environment type according to parameters
# [x] Environments: 4 doors, maze like room (call maze_1)
# [xx] make it so environment takes env_params 
# [xx] should player spawn locations and goal locations be part of the params? 
# [x] fix goal location (low priority)
# [ ] fix RR implementation.  ** LOL JUST DO THURSDAY? 
# [x] fix protocol specific code for hallway  
# [ ] formulate multichannel protocol ** DO THURSDAY
# [ ] formulate multichannel code 
# [X] test hallway code, 3 agents pre reward changes 
# [ ] test hallway code, 3 agetns POST reward changes 

 
# Questions:
# [] what should the environment tell me protocol wise
# [] do I want different environment files for different types of environments? 
# [x] I think my ideal would be to have protocols somehow inputed in in their own file and then the environment set up be different
# [x] I think a middle ground would bet to have the environment parameters in another file. 

