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
# class GridGameEnvironment(ParallelEnv):
#     def __init__(self, number_agents=2, gridsize=10, timeout=200):
#         assert type(gridsize) is tuple or int
#         self.grid_size = gridsize if type(gridsize) is tuple else (gridsize, gridsize)

class GridGameEnvironment(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, number_agents=2, size=10, timeout=200, protocol = 'CSMA_CD', render_mode=None, acting_agent = None, turn_length = None, map_type = 'hallway'):
        assert protocol == 'CSMA_CD' or protocol == 'STR' or protocol == "RR" or protocol == "NoneBlind" or protocol == "NoneSeeing", f"Protocol must be one of 'CSMA_CD', 'STR', 'RR', 'NoneBlind', 'NoneSeeing'. Got {protocol}"
        assert type(size) is tuple or int
        self.size = size
        if map_type == 'four_door' or map_type == 'four_door_space':
            if size % 2 == 0: #make the four door environment odd in size for simplicity. not actually necessary. 
                size = size + 1
        self.map_type = map_type
        self.grid_size = size if type(size) is tuple else (size, size)
        
        ## RENDERING ##
        self.window_size = 2048
        self.possible_agents = ["player_" + str(r) for r in range(number_agents)]
        agent_colors = pygame.color.THECOLORS
        self.agent_colors = {}
        for i, key in enumerate(self.possible_agents):
            # self.agent_colors[key] = random.choice(list(agent_colors.keys()))
            self.agent_colors[key] = (255, 100, 100) if i % 2 == 0 else (100, 100,255)

        self.agents = []
        self.agent_objects = {} # Dictionary, will fill in reset
        self.timestep = 0
        self._create_areas()
        self.timeout = timeout
        self.protocol = protocol

        ## ROUND ROBIN ATTRIBUTES  ##
        if (acting_agent == None) and protocol == 'RR':
            self.acting_agent = self.possible_agents[0]
        else:
            self.acting_agent = acting_agent

        if turn_length == None:
            turn_length = 2 * (self.size // 3 ) 
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
        if self.map_type == 'hallway':
            # create walls
            self.walls = []
            self.spawn_area_1 = []
            self.spawn_area_2 = []
            for x in range(self.grid_size[0]):
                for y in range(self.grid_size[1]):
                    if x < self.grid_size[0]/3: # left third is spawn area 1
                        self.spawn_area_1.append(Coordinate((x, y)))
                    elif x >= 2*self.grid_size[0]/3: # right third is spawn area 2
                        self.spawn_area_2.append(Coordinate((x, y)))
                    else:
                        if y != int(self.grid_size[1]/2):
                            self.walls.append(Coordinate((x, y))) # middle is wall, except for the hallway which is in the middle
        
        elif self.map_type == 'four_door' or self.map_type == 'four_door_space':
            #TODO: make it so there are 4 starting locations and 4 ending locations. 

            self.walls = [] 
            
            self.room_NW = []
            self.room_NE = []
            self.room_SE = []
            self.room_SW = []

            center_i = self.grid_size//2
            offset = 1
            if self.map_type == 'four_door_space':
                offset = 2
                
            self.hallway_N = [Coordinate((center_i, center_i - offset))]
            self.hallway_W = [Coordinate((center_i - offset , center_i))]
            self.hallway_S = [Coordinate((center_i, center_i + offset))]
            self.hallway_E = [Coordinate((center_i + offset, center_i))]
            self.hallways = self.hallway_N + self.hallway_E + self.hallway_W + self.hallway_S ()
            

            self.spawn_area_1 = self.room_NW
            self.spawn_area_2 = self.room_NE

            for x in range(self.grid_size[0]):
                for y in range(self.grid_size[1]):
                    if x < center_i:
                        if y < center_i:
                            self.room_NW.append(Coordinate((x, y)))
                        elif y > center_i:
                            self.room_SW.append(Coordinate((x, y)))
                        elif y == center_i:
                            if (x != center_i - offset):
                                self.walls.append(Coordinate((x, y)))
                            self.walls.append(Coordinate((x, y)))
                    elif x > center_i:
                        if y < center_i:
                            self.room_NE.append(Coordinate((x, y)))
                        elif y > center_i:
                            self.room_SE.append(Coordinate((x, y)))
                        elif y == center_i:
                            if (x != center_i + offset):
                                self.walls.append(Coordinate((x, y)))
                    elif x == center_i:
                        self.walls.append(Coordinate((x, y))) 

            self.goal_1 = [Coordinate((self.grid_size[0] - 1, self.grid_size[1] -1))] # corner of SE
            self.goal_2 = [Coordinate((0, self.grid_size[1] -1)) ] # corner of SW   
            
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
        while self.possible_agents[new_acting_player_index] not in self.agents and len(self.agents) > 0:
            new_acting_player_index = (new_acting_player_index + 1) % len(self.possible_agents)

        # Grab new player
        new_acting_player = self.possible_agents[new_acting_player_index]
        return new_acting_player

    def _get_remaining_time(self, agent_name):
        '''
        Only called if the agent is in the hallway.
        Inputs: agent name 
        Return: (int) # of states they have until they reach the target room
        '''
        if self.map_type == 'hallway':
            starting_side = self._get_starting_side()

            i = self.possible_agents.index(agent_name)
            agent_start = starting_side[i]

            if agent_start == 1: # started in room 1
                hallway_end = (2 * self.grid_size[0]/3) + 1 # first column in room 2
            elif agent_start == 2: # started in room 2
                hallway_end = (self.grid_size[0]/3) - 1 # first column in room 1

            return hallway_end - self.agent_objects[agent_name].position.x
        else:
            raise ValueError(f"{self.map_type} was not a single agent map")
    
    def _get_resource_conflicts(self):
        agents_in_conflict = []
        if self.map_type == 'hallway':
            for agent_key in self.agent_objects:
                if self.grid_size[0]/3 <= self.agent_objects[agent_key].position.x < 2*self.grid_size[0]/3:
                    agents_in_conflict.append(agent_key)
        else: raise ValueError("map conflicts not defined yet")
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

            agent_index = int(self.acting_agent[self.acting_agent.index("_")+1:])

            observations = {}

            for a in self.agents:
                current_index = int(a[a.index("_") + 1:])
                observations[a] =[
                                    self.agent_objects[a].position.x, # x pos
                                    self.agent_objects[a].position.y, # y pos
                                    # 0 if agent_index % 2 == current_index % 2 else 1 # 0 if it is your turn, 1 if it is not.
                                    0 if a == self.acting_agent else 1  # 0 if it is your turn, 1 if it is not.
                ]


        elif self.protocol == 'NoneBlind':
            # create observation for every agent
            observations = {
                a: [
                    self.agent_objects[a].position.x, # x pos
                    self.agent_objects[a].position.y, # y pos
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
        if self.map_type == 'hallway':
            return self._get_observations_single()
        else: 
            return self._get_observations_multi()

    def _get_terminations(self):
        ret = {}
        for a in self.agents:
            if self.agent_objects[a] is None:
                continue
            if  self.agent_objects[a].at_goal():
                ret[a] = True
            else:
                ret[a] = False
        return ret


    def _get_obs_reward(self):
        return 0

    def _get_rewards(self, observations, at_goal_agents):
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
        base_rewards = self._get_base_rewards(at_goal_agents)
        for a , obs in observations.items():
            rel_obs = obs[-1]
            obs_reward = self._get_obs_reward() 
            # TODO: need to get observation rewards. Will be map specific. Should not be protocol specific....(?) check. 
            # TODO: need to check how the mutlichannel setting would fit in here. 
            base_rewards[a] += obs_reward

        # need a way to translate that observation to a reward
        return base_rewards


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
        Dictionary {agent name : (goal_x, goal_y)} ??'''
        ret = {}
        for a in self.agents:
            if self.agent_objects[a] is None:
                continue
            ret[a] = (self.agent_objects[a].goal_position.x, self.agent_objects[a].goal_position.y)
        return ret

    def _get_starting_side(self):
        '''
        return: [1 0 1 0 ...  1 0] for the length of the possible agents 
        1: means agent started in self.spawn_area_1 
        2: means agent started in self.spawn_area_2
        '''
        return [1 if x % 2 == 0 else 2 for x in range(len(self.possible_agents))]
        
    def get_starting_side(self):
        return self._get_starting_side()

    def reset(self, seed = None, return_info = False, options = None):
        # reset record keeping
        self.timestep = 0
        self.agents = deepcopy(self.possible_agents)

        # Create the potential spawn points
        unused_spawn_1, unused_spawn_2 = deepcopy(self.spawn_area_1), deepcopy(self.spawn_area_2)

        # spawn agents. Remove possible spawn location from list as we go so we do not spawn 2 agents on top of each other
        self.agent_objects = {}

        for i, key in enumerate(self.possible_agents):
            if i % 2 == 0:
                pos = unused_spawn_1.pop(random.randrange(len(unused_spawn_1))) # fetches a random spot and removes from list
                if self.map_type == 'four_door' or self.map_type == 'four_door_space':
                    goal = self.goal_1
                else:
                    goal = Coordinate((self.grid_size[0]-1, self.grid_size[1]-1)) # some spot in the other room. IDK where but it will be consistent
                self.agent_objects[key] = Player(position=pos,
                                                min_position=Coordinate((0,0)),
                                                max_position=Coordinate((self.grid_size[0]-1, self.grid_size[1]-1)),
                                                goal_position=goal,
                                                walls=self.walls)
            else:
                pos = unused_spawn_2.pop(random.randrange(len(unused_spawn_2))) # fetches a random spot and removes from list
                if self.map_type == 'four_door' or self.map_type == 'four_door_space':
                    goal = self.goal_2
                else:
                    goal = Coordinate((0, self.grid_size[1]-1)) # some spot in the other room. IDK where but it will be consistent
                self.agent_objects[key] = Player(position=pos,
                                                min_position=Coordinate((0,0)),
                                                max_position=Coordinate((self.grid_size[0]-1, self.grid_size[1]-1)),
                                                goal_position=goal,
                                                walls=self.walls)

        # render if human mode
        if self.render_mode == "human":
            self._render_frame()

        return self._get_observations()

    def step(self, actions):
        # execute actions for all agents
        # dont let them move to other agent positions
        for key in actions:
            agent_positions = [self.agent_objects[key].position for key in self.agents]
            self.agent_objects[key].step(actions[key], agent_positions)

        # check terminations, fetch obs. If terminated, delete agent so it does not appear in next state
        self.timestep += 1
        terminations = self._get_terminations()
        ## PSEDUO CODE FOR SOPHIA ##
        # Q: what is get infos?
        # Q: what are truncations. 
        # Q: can you now do the whole RR thing? 
        ######### MY CODE ###########
        at_goal_agents = []
        for key in self.agent_objects:
            if self.agent_objects[key].at_goal() and key in self.agents:
                self.agents.remove(key)
                at_goal_agents.append(key)
        truncations = {a:self.timestep >= self.timeout for a in self.agents}
        observations = self._get_observations() 
        infos = self._get_infos() 
        rewards = self._get_rewards(observations, at_goal_agents)
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
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the walls
        for w in self.walls:
            pygame.draw.rect(
                canvas,
                (128, 128, 128), # grey
                pygame.Rect(
                    (pix_square_size * w.x, pix_square_size * w.y),
                    (pix_square_size, pix_square_size),
                ),
            )

        # IDEALLY want to color the goal squares.
        img = pygame.transform.scale(pygame.image.load('env/star.png'), (pix_square_size, pix_square_size))
        destinations = [(0, self.grid_size[1] - 1), (self.grid_size[0] - 1, self.grid_size[1] - 1)]
        for x,y in destinations:
            canvas.blit(img, (x*pix_square_size, y*pix_square_size))


        # color the bottlenecks
        bottleneck_squares = [(x , int(self.grid_size[1]/2)) for x in range(self.grid_size[0]//3,  math.ceil(2*self.grid_size[0]/3)+1)]
        for square in bottleneck_squares:
            bottom_x, bottom_y = square
            top_x, top_y = bottom_x+1, bottom_y+1
            bottom_x*= pix_square_size
            bottom_y*= pix_square_size
            top_x*= pix_square_size
            top_y *= pix_square_size
            number_lines = 5
            width_x = (top_x-bottom_x)/number_lines
            width_y = (top_y-bottom_y)/number_lines
            for i in range(1,number_lines+1):
                start = (bottom_x, bottom_y + i*width_y)
                end = (bottom_x + i*width_x, bottom_y)
                pygame.draw.line(canvas, (255,0,255), start, end, width=5)

                start = (top_x,  top_y-i*width_x)
                end = (top_x - i*width_x, top_y)
                pygame.draw.line(canvas, (255,0,255), start, end, width=5)
        # Now we draw the agent
        # for count, val in enumerate(self.agent_objects.values()):
        for i, key in enumerate(self.agents):
            
            center = (pix_square_size * (self.agent_objects[key].position.x + 0.5),
                      pix_square_size * (self.agent_objects[key].position.y + 0.5))
            pygame.draw.circle(
                canvas,
                self.agent_colors[key],
                center,
                pix_square_size / 2.1,
            )
            s = key[key.find("_")+1:]
            my_font = pygame.font.SysFont('Comic Sans MS', self.window_size//(1*self.size))
            x,y = my_font.size(s)
            text_surface = my_font.render(s, True, 'black')
            # canvas.blit(text_surface, (center[0]-width/2 - offset, center[1]-height/2))
            canvas.blit(text_surface, (center[0]-x/2, center[1]-y/2))

    
         
        # Finally, add some gridlines
        for x in range(self.size + 1):
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
        return MultiDiscrete([self.grid_size[0], self.grid_size[1], 2]) # consists of X pos, Y pos, hallway occupied or not
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4) # consists of Up (0), Right (1), Down (2), Left (3)