from env.player import Coordinate
class env_params():
    def __init__(self, number_agents,  size=10, env_type = "hallway", ):
        '''
        This is supposed to just set up the grid and protocol regions etc. 
        '''
        if env_type == 'four_door' or env_type == 'four_door_space':
            if size % 2 == 0: #make the four door environment odd in size for simplicity. not actually necessary. 
                size = size + 1

        self.size = size
        self.grid_size = size if type(size) is tuple else (size, size)
        self.window_size = 512
        self.env_type = env_type
        self.possible_agents = ["player_" + str(r) for r in range(number_agents)]
        
        # agent_colors = pygame.color.THECOLORS
        # self.agent_colors = {}
        # for key in self.possible_agents:
            # self.agent_colors[key] = random.choice(list(agent_colors.keys()))

        # self.agents = []
        # self.agent_objects = {} # Dictionary, will fill in reset
        # self.timestep = 0
        self._create_areas()

        # self.timeout = timeout
        # self.protocol = protocol

    def create_areas(self):
        '''
        This should define spawn areas, walls, starting areas, goal locations, protocol zones
        '''
        if self.env_type == 'hallway':
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
            
            self.hallway = []
            # TODO: build hallway
            # TODO: get goal locations

        elif self.env_type == 'four_door' or self.env_type == 'four_door_space':
            #TODO: make it so there are 4 starting locations and 4 ending locations. 

            self.walls = [] 
            
            self.room_NW = []
            self.room_NE = []
            self.room_SE = []
            self.room_SW = []

            center_i = self.grid_size//2
            offset = 1
            if self.env_type == 'four_door_space':
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
                 
        
        elif self.env_type == 'maze_1':
            pass
    
    def get_starts(self):
        for agent in self.possible_agents:
            
            
        pass

    def get_goals(self):
        pass