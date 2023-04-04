import random
from typing import List

import numpy
from pettingzoo.utils.env import ParallelEnv
from copy import copy, deepcopy
import functools
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np
from .player import Coordinate
from .player import Player
import pygame
import math
# class DroneGameEnvironment(ParallelEnv):
#     def __init__(self, number_agents=2, gridsize=10, timeout=200):
#         assert type(gridsize) is tuple or int
#         self.grid_size = gridsize if type(gridsize) is tuple else (gridsize, gridsize)

class DroneGameEnvironment(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, number_agents=6, size=5, timeout=200, protocol = 'CSMA_CD', render_mode=None, acting_agent = None, turn_length = None,):
        assert protocol == 'CSMA_CD' or protocol == 'STR' or protocol == "RR" or protocol == "NoneBlind" or protocol == "NoneSeeing", f"Protocol must be one of 'CSMA_CD', 'STR', 'RR', 'NoneBlind', 'NoneSeeing'. Got {protocol}"
        assert type(size) is tuple or int
        # self.size = size
        #self.grid_size = size if type(size) is tuple else (size, size)
        self.grid_size = (6,6)

        ## RENDERING ##
        self.window_size = 2048
        self.possible_agents = ["player_" + str(r) for r in range(number_agents)]
        agent_colors = pygame.color.THECOLORS
        self.agent_colors = {}
        for i, key in enumerate(self.possible_agents):
            # self.agent_colors[key] = random.choice(list(agent_colors.keys()))
            self.agent_colors[key] = (255, 100, 100) #  if i % 2 == 0 else (100, 100,255)

        self.agents = []
        self.agent_objects = {} # Dictionary, will fill in reset
        self.batteries = {}
        self.timestep = 0
        self._create_areas()
        self.timeout = timeout
        self.protocol = protocol
        self.max_battery = 30
        self.charging_time = 1
        self.time_charging = {}

        ## ROUND ROBIN ATTRIBUTES  ##
        if (acting_agent == None) and protocol == 'RR':
            self.acting_agent = self.possible_agents[0]
        else:
            self.acting_agent = acting_agent

        if turn_length == None:
            turn_length = 1 # 2 * (self.size // 3 )
        self.turn_length = turn_length
        self.turn_time = turn_length

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    # create different areas of interest.
    # Left third is spawn area 1
    # right third is spawn area 2
    # middle third is a wall, except for the hallway which is in the middle

    
    def _create_areas(self):
        self.recharging_stations = []
        self.job_stations = []
        center_x = self.grid_size[0] // 2
        center_y = self.grid_size[1] // 2
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if center_x-1 <= i <= center_x and center_y-1 <= j <= center_y:
                    self.recharging_stations.append((i,j))
                if i < 1 or i >= self.grid_size[0] - 1 or j < 1 or j >= self.grid_size[1] - 1:
                    self.job_stations.append((i,j))


        

            
    ### PROTOCOL (AND MAP) SPECIFIC ###      
    def _get_next_player_RR(self):
        '''
        Ideas to improve:
        **Make sure we are skipping agents once they are no longer active
        *Have the individual's turn end once they are through the hallway? "Network job" complete and all
        more *s = more important 
        '''
        # Find index of current player
        acting_player_index = self.possible_agents.index(self.acting_agent)

        # Go to the next (active) player index
        new_acting_player_index = (acting_player_index + 1) % len(self.possible_agents)

        # Grab new player
        new_acting_player = self.possible_agents[new_acting_player_index]
        return new_acting_player

    def _get_remaining_time(self, agent_name):
        '''
        Only called if the agent is in the hallway.
        Inputs: agent name 
        Return: (int) # of states they have until they reach the target room
        '''
        return 1 # all agents take 1 timestep alone to recharge
    
    def _get_resource_conflicts(self):
        agents_in_conflict = []
        for agent_key in self.agent_objects:
            pos = self.agent_objects[agent_key].position.x, self.agent_objects[agent_key].position.y
            if pos in self.recharging_stations:
                agents_in_conflict.append(agent_key)
        return agents_in_conflict
                
    ## SORT OUT NECCECARY QUERIES ###
    def _get_map_queries(self, conflicts = True, time_remaining = False):
        if time_remaining:
            assert conflicts 
        time_remaining_list = []
        if conflicts:
            agents_in_conflict = self._get_resource_conflicts()
        if time_remaining: 
            for agent_key in agents_in_conflict:
                time_to_go = self._get_remaining_time(agent_key)
                time_remaining_list.append(time_to_go)
        return agents_in_conflict, time_remaining_list   
        

    ### FUNCTION SPECIFIC ###
    def _get_observations_single(self):
        '''
        We are going to need different observations for different protocols
        This covers the case any environment with a single conflict region. 
        '''
        # calculate if any agent is in the hallway
        if self.protocol == 'CSMA_CD':  
            agents_in_shared_resource, _ = self._get_map_queries(conflicts = True, time_remaining = False)
            # create observation for every agent
            observations = {
                a: [
                    self.agent_objects[a].position.x, # x pos
                    self.agent_objects[a].position.y, # y pos
                    1 if self.batteries[a] > 0 else 0 , # whether or not we have battery
                    0 if len(agents_in_shared_resource) == 0 or len(agents_in_shared_resource) == 1 and a in agents_in_shared_resource else 1  #shared resource occupied
                    ]
                for a in self.agents
            }

        elif self.protocol == 'STR':
            self.turn_time = max(0, self.turn_time - 1) # time floor at 0

            # find how many agents in the shared and how far they are from finishing with the resource
            agents_in_shared_resource, time_remaining = self._get_map_queries(conflicts = True, time_remaining = True)

            # Case: it is no one's turn and if there is a conflict. 

            if (self.turn_time == 0) and (len(agents_in_shared_resource) >= 1):
                # print("time_remaining", time_remaining)
                # print("agents in shared resource", agents_in_shared_resource)
                winning_index = time_remaining.index(max(time_remaining))
                winning_agent = agents_in_shared_resource[winning_index]
                if len(agents_in_shared_resource) == 1: 
                    # if they are the only agent, do not start the turn timer yet or assign right of way, wait for conflict to do that. 
                    self.turn_time = 0
                    self.acting_agent = None
                else:
                    try: 
                        self.turn_time = time_remaining[winning_index] + 2  # Set the length of chosen agent's turn   
                        self.acting_agent = winning_agent 
                    except: 
                        print("winning index", winning_index)
                        print("agents in shared resource", agents_in_shared_resource)


            # Case: it is already a players turn, let them continue. It doesn't matter where they are compared to the other players.
            elif self.turn_time > 0:
                winning_agent = self.acting_agent

            # Case: it is no one's turn and no one is trying the hallway. Keep waiting
            elif (self.turn_time == 0) and len(agents_in_shared_resource) == 0:
                winning_agent = None
                self.acting_agent = None

            observations = {
                a: [
                    self.agent_objects[a].position.x, # x pos
                    self.agent_objects[a].position.y, # y pos
                    1 if self.batteries[a] > 0 else 0 , # whether or not we have battery
                    0 if len(agents_in_shared_resource) == 0 or a == winning_agent else 1  # 0 = player's turn in the hallway, 1 =  hallway occupied
                    ]
                for a in self.agents}  

        elif self.protocol == 'RR': # RR has NO map specific information. 
            if self.turn_time == 0:
                next_player = self._get_next_player_RR()
                self.acting_agent = next_player
                self.turn_time = self.turn_length
            else:
                self.turn_time = self.turn_time - 1
                
            observations = {
                a: [
                    self.agent_objects[a].position.x, # x pos
                    self.agent_objects[a].position.y, # y pos
                    1 if self.batteries[a] > 0 else 0 , # whether or not we have battery
                    0 if a == self.acting_agent else 1 # 0 if it is your turn, 1 if it is not.
                    ]
                for a in self.agents
            }

        elif self.protocol == 'NoneBlind':
            # create observation for every agent
            observations = {
                a: [
                    self.agent_objects[a].position.x, # x pos
                    self.agent_objects[a].position.y, # y pos
                    1 if self.batteries[a] > 0 else 0 , # whether or not we have battery
                    0
                    ]
                for a in self.agents }

        elif self.protocol == "NoneSeeing":
            agents_in_shared_resource, _ = self._get_map_queries(conflicts=True, time_remaining=False)
            # create observation for every agent
            observations = {
                a: [
                    self.agent_objects[a].position.x,  # x pos
                    self.agent_objects[a].position.y,  # y pos
                    1 if self.batteries[a] > 0 else 0 , # whether or not we have battery
                    0 if len(agents_in_shared_resource) == 0 or len(
                        agents_in_shared_resource) == 1 and a in agents_in_shared_resource else 1
                    # shared resource occupied
                ]
                for a in self.agents
            }
        else: 
            raise ValueError("Protocol not defined yet")
        
        return observations
    
    def _get_observations_multi(self):
        raise ValueError("multi channel protocols not implemented yet")

    def _get_observations(self):
        return self._get_observations_single()


    def _get_terminations(self):
        return {a:False for a in self.agents}


    def _get_rewards(self, observations):
        '''
        This function returns the rewards for a protocol and a map. Weird. We don't really like that. 
        Honestly ideally we would make this take in observations and map type and tell us the reward I feel like? ya?
        Should think of this as we go on to multichannel methods. base reward + 1 for each channel. 
        [] problem: cannot use observations since we do rewards, the delete "completed" agents, then get observations. (bad). 
        [] this means that there are some agents that should get reward but DO NOT have rewards. (the agents that win that time only though.. )
        [] well then we should give them R 1 if they are removed that time and 0 else. 
        Returns: {agent: reward for agent in self.agents}
        '''
        
        # TODO: hype up
        rewards = {}
        for a , obs in observations.items():
            rewards[a] = 1.0 if (obs[0], obs[1]) in self.job_stations and obs[2] > 0 else 0.0

        # need a way to translate that observation to a reward
        return rewards


    def _get_base_rewards(self, at_goal_agents):
        # biden is a goofy guy haha running across the lawn in the rain
        ''' 
        returns the base reward: if agent is in a waiting phase
        '''
        ret = {a:-0.01  for a in self.agents}
        for agent in at_goal_agents:
            ret[agent] = 1.0
        return ret
        

    def _get_infos(self):
        '''
        Dictionary {agent name : (goal_x, goal_y)} ??
        '''
        return {a: (self.agent_objects[a].goal_position.x, self.agent_objects[a].goal_position.y) for a in self.agents}


    def reset(self, seed = None, return_info = False, options = None):
        # reset record keeping
        self.timestep = 0
        self.agents = deepcopy(self.possible_agents)
        self.batteries = {a:numpy.random.randint(0, self.max_battery) for a in self.agents}
        self.time_charging = {a:0 for a in self.agents}

        # Create the potential spawn points
        unused_spawn = [Coordinate((i,j)) for i in range(self.grid_size[0]) for j in range(self.grid_size[1])]

        # spawn agents. Remove possible spawn location from list as we go so we do not spawn 2 agents on top of each other
        self.agent_objects = {}
        for i, key in enumerate(self.possible_agents):
                pos = unused_spawn.pop(random.randrange(len(unused_spawn))) # fetches a random spot and removes from list
                goal = Coordinate((self.grid_size[0]-1, self.grid_size[1]-1)) # some spot in the other room. IDK where but it will be consistent
                self.agent_objects[key] = Player(position=pos,
                                                min_position=Coordinate((0,0)),
                                                max_position=Coordinate((self.grid_size[0]-1, self.grid_size[1]-1)),
                                                goal_position=goal,
                                                 walls=[])

        # render if human mode
        if self.render_mode == "human":
            self._render_frame()
        return self._get_observations()

    def step(self, actions):
        # execute actions for all agents
        # dont let them move to other agent positions
        for key in actions:
            agent_positions = [self.agent_objects[key].position for key in self.agents]
            # if self.batteries[key] > 0:
            self.agent_objects[key].step(actions[key], agent_positions)
            self.batteries[key] = max(0, self.batteries[key] - 1)

        agents_in_charging = self._get_resource_conflicts()
        if len(agents_in_charging) != 1:
            self.time_charging = {a: 0 for a in self.agents}
        else:
            self.time_charging[agents_in_charging[0]] += 1
            if self.time_charging[agents_in_charging[0]] >= self.charging_time:
                self.batteries[agents_in_charging[0]] = self.max_battery

        # check terminations, fetch obs. If terminated, delete agent so it does not appear in next state
        self.timestep += 1
        terminations = self._get_terminations()
        ## PSEDUO CODE FOR SOPHIA ##
        # Q: what is get infos?
        # Q: what are truncations. 
        # Q: can you now do the whole RR thing? 
        ######### MY CODE ###########
        truncations = {a:self.timestep >= self.timeout for a in self.agents}
        observations = self._get_observations() 
        infos = self._get_infos() 
        rewards = self._get_rewards(observations)
        ########### old code #################
        # rewards = self._get_rewards()
        # # now delete agents at destination
        # for key in self.agent_objects:
        #     if self.agent_objects[key].at_goal() and key in self.agents:
        #         self.agents.remove(key)

        # truncations = {a:self.timestep >= self.timeout for a in self.agents}
        # observations = self._get_observations() 
        # infos = self._get_infos() 

        # render if human mode
        if self.render_mode == "human":
            self._render_frame()

        return observations, rewards, terminations, truncations, infos

    """ Rendering """
    def render(self):
        return self._render_frame()

    def _render_frame(self):
        # init rendering
        if self.window is None and self.render_mode is not None:
            pygame.init()
            pygame.font.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.window_size, self.window_size)
                )
        if self.clock is None and self.render_mode is not None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / max(self.grid_size)
        )  # The size of a single grid square in pixels

        # First we draw the charging stations
        for cs in self.recharging_stations:
            pygame.draw.rect(
                canvas,
                (128, 255, 128), # light green
                pygame.Rect(
                    (pix_square_size * cs[0], pix_square_size * cs[1]),
                    (pix_square_size, pix_square_size),
                ),
            )

        # Next color tasks squares
        for js in self.job_stations:
            pygame.draw.rect(
                canvas,
                (128, 128, 255), # light blue
                pygame.Rect(
                    (pix_square_size * js[0], pix_square_size * js[1]),
                    (pix_square_size, pix_square_size),
                ),
            )



        # Now we draw the agents
        for i, key in enumerate(self.agents):

            size = 0.7
            img = pygame.transform.scale(pygame.image.load('env/drone.png'), (pix_square_size*size, pix_square_size*size))
            canvas.blit(img, ((self.agent_objects[key].position.x + (1-size)/2)* pix_square_size, (self.agent_objects[key].position.y + (1-size)/2) * pix_square_size))

            # also add charge to show full state in image
            if self.batteries[key] > 0:
                img = pygame.transform.scale(pygame.image.load('env/lightning.png'),
                                             (pix_square_size * (1-size)/2, pix_square_size * (1-size)/2))
                canvas.blit(img, ((self.agent_objects[key].position.x + 0.05) * pix_square_size,
                                  (self.agent_objects[key].position.y + 0.05) * pix_square_size))
            else:
                img = pygame.transform.scale(pygame.image.load('env/lightning_crossed.png'),
                                             (pix_square_size * (1 - size) / 2, pix_square_size * (1 - size) / 2))
                canvas.blit(img, ((self.agent_objects[key].position.x + 0.05) * pix_square_size,
                                  (self.agent_objects[key].position.y + 0.05) * pix_square_size))

    
         
        # Finally, add some gridlines
        for x in range(self.grid_size[0] + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        pass
        # if self.window is not None:
        #     pygame.display.quit()
        #     pygame.quit()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete([self.grid_size[0], self.grid_size[1], 2, 2]) # consists of X pos, Y pos, battery charge, protocol state
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4) # consists of Up (0), Right (1), Down (2), Left (3)