import random

import gym.wrappers.monitoring.video_recorder
import gymnasium.spaces
import numpy as np
import numpy.random

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
from PIL import Image
from env.dronegame import DroneGameEnvironment

env = GridGameEnvironment(number_agents=15, size=10,  render_mode="rgb_array")
# env = DroneGameEnvironment(render_mode="rgb_array")
obs = env.reset()
img = env.render()

img = Image.fromarray(img)
img.save("screenshot.jpg")



# run test
# env = DroneGameEnvironment(number_agents=15, size=7, protocol="NoneBlind",  render_mode="human")
obs = env.reset()
for j in range(100):
        actions = {a:numpy.random.randint(0, 3+1) for a in obs}
        obs, rewards, dones, truncations, infos = env.step(actions)
        print(obs["player_0"], rewards["player_0"])
