from copy import deepcopy
from enum import Enum
import numpy as np
import random


# defines a coordinate system
# Just a pair of x,y values
# but will be convenient for checking if we are in a wall or not

class Coordinate:
    def __init__(self, position_tuple):
        assert type(position_tuple) is tuple
        self.x = position_tuple[0]
        self.y = position_tuple[1]

    # define easy handles for moving a coordinate in any direction
    def up(self):
        self.y+=1
    def down(self):
        self.y-=1
    def left(self):
        self.x-=1
    def right(self):
        self.x+=1

    # define comparisons for convenience
    def __eq__(self, other):
        assert type(other) is Coordinate
        return self.x == other.x and self.y == other.y
    def __lt__(self, other):
        assert type(other) is Coordinate
        return self.x < other.x and self.y < other.y
    def __le__(self, other):
        assert type(other) is Coordinate
        return self.x <= other.x and self.y <= other.y
    def __gt__(self, other):
        assert type(other) is Coordinate
        return self.x > other.x and self.y > other.y
    def __ge__(self, other):
        assert type(other) is Coordinate
        return self.x >= other.x and self.y >= other.y

    # define how to show it in prompt
    def __str__(self):
        return "({},{})".format(self.x, self.y)

# Follows Compass style ordering style.
# Simply provides a word for directions instead of some random number
class Moves(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

# Describes a agent in a grid world. Has a X,Y position, and limits on how far it can go in each direction
# Optionally has a goal position, and a function to check if it is at the goal
# Also has walls that it cannot enter.
class Player:

    def __init__(self, position, max_position, min_position, goal_position, walls):
        '''
        actions = [0, 1, 2, 3]
        '''
        # print("I am here 1")
        assert type(position) is Coordinate, "Agents position must be a Coordinate, got {}".format(type(position))
        assert type(max_position) is Coordinate, "Agents max_position must be a Coordinate, got {}".format(type(max_position))
        assert type(min_position) is Coordinate, "Agents min_position must be a Coordinate, got {}".format(type(min_position))
        assert type(goal_position) is Coordinate, "Agents goal_position must be a Coordinate, got {}".format(type(goal_position))

        self.position = position
        self.max_position = max_position
        self.min_position = min_position
        self.goal_position = goal_position
        self.walls = walls

        #self.actions = actions
        #self.q_table = np.zeros([self.max_position.x, self.max_position.y, len(self.actions)])

    # Moves the agent according to the move given. Move must be in the set (0,1,2,3).
    # does not move out of bounds. If it were to go out of bounds, this is a no op.
    
    def step(self, move, agent_positions):
        current_position = deepcopy(self.position)

        # move current position
        if      move == Moves.UP.value:       current_position.up()
        elif    move == Moves.RIGHT.value:    current_position.right()
        elif    move == Moves.DOWN.value:     current_position.down()
        elif    move == Moves.LEFT.value:     current_position.left()
        else:   raise Exception("Move must be one of 0,1,2,3")

        # verify current position in bounds
        in_bounds = self.min_position <= current_position <= self.max_position

        # verify current position not in a wall
        in_wall = current_position in self.walls

        # verify not blocked by other agent
        in_agent = current_position in agent_positions

        # change current position if in bounds and not in a wall
        # else, we do not move
        if in_bounds and not in_wall and not in_agent:
            self.position = current_position

    # def get_next_action(self, learning_params):
        
    #     print("I am here")
    #     T = learning_params.T

    #     pr_sum = np.sum(np.exp(self.q[self.position.x, self.position.y :] * T))
    #     pr = np.exp(self.q[self.position.x,self.position.y, :] * T)/pr_sum # pr[a] is probability of taking action a

    #     # If any q-values are so large that the softmax function returns infinity, 
    #     # make the corresponding actions equally likely

    #     if any(np.isnan(pr)):
    #         print('BOLTZMANN CONSTANT TOO LARGE IN ACTION-SELECTION SOFTMAX.')
    #         temp = np.array(np.isnan(pr), dtype=float)
    #         pr = temp / np.sum(temp)

    #     pr_select = np.zeros(len(self.actions) + 1)
    #     pr_select[0] = 0
    #     for i in range(len(self.actions)):
    #         pr_select[i+1] = pr_select[i] + pr[i]

    #     randn = random.random()
    #     for a in self.actions:
    #         if randn >= pr_select[a] and randn <= pr_select[a+1]:
    #             a_selected = a
    #             break

    #     return a_selected

    # def update_q_function(self, position, position_new, action, reward, learning_params):
    #     alpha = learning_params.alpha
    #     gamma = learning_params.gamma

    #     # Bellman update
    #     self.q_table[position_new.x][position_new.y][action] = (1-alpha)*self.q_table[position.x][position.y][action] + alpha*(reward + gamma * np.amax(self.q_table[position_new.x][position_new.y]))

    # def update_agent(self,action, reward, learning_params): 
    #     old_position = self.position
    #     self.step(action)
    #     new_position = self.position
    #     self.update_q_function(old_position, new_position, action, reward, learning_params)

    # Returns true if its current position equals its goal position, false otherwise
    def at_goal(self):
        # if self.position == self.goal_position: 
            # print("I am at goal!, position: " , self.position)
        return self.position == self.goal_position

    def in_hallway(self):
        return self.max_position.x/3 <= self.position.x < 2*self.max_position.x/3