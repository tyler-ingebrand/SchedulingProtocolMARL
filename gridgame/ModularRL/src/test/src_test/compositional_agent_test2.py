import math

import gym
from gym.wrappers import FlattenObservation
from src import *

# env generation:
# Use FlattenObservation on dict env, such as 'FetchPickAndPlace-v1':
env = FlattenObservation(gym.make('FetchPickAndPlace-v1'))
steps = 1000_000

# define agents
agent_grab = PPO_Agent("MlpPolicy", # one of "MlpPolicy", "CnnPolicy"
                      env.observation_space,
                      env.action_space,
                      hook = Reward_Per_Episode_Hook()
                 )
agent_move = PPO_Agent("MlpPolicy", # one of "MlpPolicy", "CnnPolicy"
                      env.observation_space,
                      env.action_space,
                      hook = Reward_Per_Episode_Hook()
                 )
def determine_active(state): # only valid for fetch pick and place
    # difference in arm and obj position
    x = state[6] - state[9]
    y = state[7] - state[10]
    z = state[8] - state[11]
    if math.sqrt(x**2 + y**2 + z**2) > 0.026:
        return 0
    else:
        return 1

# helper
def CartesianDistance(x,y):
    sum_ = 0
    for x_i, y_i in zip(x,y):
        sum_ += (x_i - y_i)**2
    return math.sqrt(sum_)
def DotProduct(x,y):
    sum_ = 0
    for x_i, y_i in zip(x, y):
        sum_ += x_i * y_i
    return sum_
def reward_grab(s,a,ns):
    arm_pos = (s[6], s[7], s[8])
    obj_pos = (s[9], s[10], s[11])
    obj_vel = (s[20], s[21], s[22])
    arm_vel = (s[26], s[27], s[28])
    obj_rot = (s[17], s[18], s[19])
    action = (a[0], a[1], a[2])

    # compute if we are going the right way or not
    directionToDestination = (obj_pos[0] - arm_pos[0], obj_pos[1] - arm_pos[1], obj_pos[2] - arm_pos[2],) # points from arm to obj
    correct_direction = DotProduct(directionToDestination, action) / (CartesianDistance(directionToDestination, (0,0,0)) * CartesianDistance(action, (0,0,0)) )
    if math.isnan(correct_direction):
        correct_direction = 0.5
    correct_direction_reward = (-1 + correct_direction)/2 # between -1 and 0


    # punish if we move fast by box. This leads us to accidentally hit the box gonig fast and move it
    # Take our velocity and divide by distance to destination. if that distance is large, then moving fast is punished slightly.
    # if that distance is small, moving fast is punished more
    # punishment_close_vel = max(-Magnitude(arm_vel) / (10 * Magnitude(arm_pos - obj_pos) + 0.1), -1)
    punishment_close_vel = -CartesianDistance(arm_vel, (0,0,0))


    # reward done
    gripper_pos = s[15] + s[16]
    if CartesianDistance(arm_pos, obj_pos) < 0.026 and gripper_pos < 0.05:
        reward_done = 10
    elif obj_pos[2] < 0.2:
        reward_done = -30
    else:
        reward_done = 0
    return correct_direction_reward + punishment_close_vel + reward_done


def reward_move(s,a,ns):
    arm_pos = (s[6], s[7], s[8])
    obj_pos = (s[9], s[10], s[11])
    obj_vel = (s[20], s[21], s[22])
    arm_vel = (s[26], s[27], s[28])
    obj_rot = (s[17], s[18], s[19])
    action = (a[0], a[1], a[2])
    dest_pos = (s[3], s[4], s[5])

    reward_done = 100.0 if CartesianDistance(dest_pos, obj_pos) < 0.01  else -100.0 if obj_pos[2] < 0.2  else 0.0
    distance_reward = -5 * CartesianDistance(dest_pos, obj_pos)
    return  reward_done + distance_reward

agent = Compositional_Agent([agent_grab, agent_move],
                            determine_active,
                            [reward_grab, reward_move],
                            hook=Reward_Per_Episode_Hook()
                            )

run(env, agent, steps=steps)

# see results
agent.plot()
run(env, agent, 10_000, render=True, train=False)
