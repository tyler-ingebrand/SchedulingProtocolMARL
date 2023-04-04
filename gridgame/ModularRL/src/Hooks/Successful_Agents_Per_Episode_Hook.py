import matplotlib.pyplot as plt
from src.Hooks.Abstract_Hook import Abstract_Hook
import numpy

# A hook which keeps track of reward per episode in training
# This is in training mode only, no evaluation is performed. Also this only works for 1 env at a time
# Also keeps track of number of steps and episodes observed
# Outputs a dict containing "episode_rewards", "n_steps", "n_episodes"

class Successful_Agents_Per_Episode_Hook(Abstract_Hook):
    def __init__(self):

        self.current_episode_num_agents = 0
        # self.rewards = []
        self.num_successful_agents = []
        self.number_steps = 0
        self.number_episodes = 0

    def observe(self, agent, obs, action, reward, done, truncated, info, tag = "1"):
        '''
        Counts the number of agents that get a reward of 1 during an training episode. 
        Records this over all episodes
        Inputs 
            reward: (dict) {agent_name: agent reward} '''
        # what this to record the number of agents that succeed by the end of each episode 
        # It CAN record when agents complete their mission but that seems a bit overkill. 
        # Another thing that could be good to record is -> the number of steps until the last agent completes the mission?
        # or else the average number of steps for the agents to complete the mission. (max steps for failed) -> this one could weird? (might have a lower variance than the no protocol) 
        
        for agent_id, agent_reward in reward.items():
            # ONLY COLLECTING # OF AGENTS TO REACH GOAL. does not record which agents or when
            if agent_reward == 1.0:
                self.current_episode_num_agents = self.current_episode_num_agents + 1  #how am i increasing number of agents. 
             
        self.number_steps += 1
        # DONE AND TRUNCATED MIGHT BE DICTIONARIES OPE! 
        # print("done is", done)
        # print("truncated is", truncated)
        # look at run.py lines 85 - 94
        #terminations = done , truncations = truncated

        if all(done.values()) or all(truncated.values()) or len(done) == 0 or len(truncated) == 0:
            self.number_episodes += 1
            self.num_successful_agents.append(self.current_episode_num_agents)
            self.current_episode_num_agents = 0

        # if done or truncated:
        #     self.number_episodes += 1
        #     self.num_successful_agents.append(self.current_episode_num_agents)
        #     self.current_episode_num_agents = 0

    def get_output(self):
        return {"episode_rewards" : self.rewards,
                "n_steps" :         self.number_steps,
                "n_episodes":       self.number_episodes}

    def plot(self):
        plt.plot(self.num_successful_agents)
        plt.title("Number of Successful Agents per Episode")
        plt.xlabel("Episodes")
        plt.show()
    
