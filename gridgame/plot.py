import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

results_dir = "results"
protocols = ['CSMA_CD', 'STR', 'RR', 'NoneBlind', 'NoneSeeing']
seeds = 5
number_agents = 15


headers = []
columns = []
for p in protocols:

    # for a given protocol, get all agent success lists
    protocol_successes = []
    for s in range(seeds):
        file_name = f"{results_dir}/{number_agents}_agents_{p}_seed_{s}.pkl"
        try:
            if os.path.getsize(file_name) > 0:
                with open(file_name , 'rb') as f :
                    protocol_hook = pickle.load(f)
            else:
                continue

        except:
            continue
        agent_successes = protocol_hook.num_successful_agents
        agent_successes = agent_successes[:25_000]
        agent_successes = [100*a/number_agents for a in agent_successes] # scale to percent
        protocol_successes.append(agent_successes)

    if len(protocol_successes) == 0:
        continue
    # compute averages
    low, high, median = [], [], []
    for episode_i_successes in zip(*protocol_successes):
        ep_quartiles = np.quantile(episode_i_successes, [0.25, 0.5, 0.75])
        low.append(ep_quartiles[0])
        median.append(ep_quartiles[1])
        high.append(ep_quartiles[2])

    # add data to csv
    episodes = len(median)
    window = episodes//100
    low = [np.mean(low[i: i+window]) for i in range(0, episodes, window)]
    median = [np.mean(median[i: i+window]) for i in range(0, episodes, window)]
    high = [np.mean(high[i: i+window]) for i in range(0, episodes, window)]
    episode_indicies = [i for i in range(0, episodes, window)]

    # add all to csv stuff
    headers.append(f"{p}_episodes")
    headers.append(f"{p}_low")
    headers.append(f"{p}_median")
    headers.append(f"{p}_high")
    columns.append(episode_indicies)
    columns.append(low)
    columns.append(median)
    columns.append(high)

    # plot everything
    plt.plot(episode_indicies, median, label=p)
    plt.fill_between(x=episode_indicies,
                     y1=low,
                     y2=high,
                     alpha=0.5)

# add plot labels and save
plt.xlabel("Training Episode")
plt.ylabel("Successful Agent %")
plt.title("Concurrent Training Performance")
plt.legend(loc="best")
plt.savefig(f'{results_dir}/graph.png')


# create csv
import csv
with open(f'{results_dir}/graph.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(headers)
    for results in zip(*columns):
        spamwriter.writerow(results)