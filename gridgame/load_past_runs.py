import pickle
import matplotlib.pyplot as plt
import numpy as np
import os  

no_protocol_file_name = "results/15_agents_hook_No-Protocol.pkl"
protocol_file_name = 'results/15_agents_hook_CSMA-CD-Protocol.pkl'
STR_protocol_file_name = 'results/good_hooks/15_agents_hook_STR-Protocol_quality.pkl'
RR_protocol_file_name = 'results/good_hooks/15_agents_hook_RR-Protocol_200_timeout.pkl'

if os.path.getsize(protocol_file_name) > 0:  
    with open(protocol_file_name , 'rb') as f :
        protocol_hook = pickle.load(f)

if os.path.getsize(no_protocol_file_name) > 0:  
    with open(no_protocol_file_name , 'rb') as f :
        no_protocol_hook = pickle.load(f)

if os.path.getsize(STR_protocol_file_name) > 0:  
    with open(STR_protocol_file_name , 'rb') as f :
        STR_protocol_hook = pickle.load(f)

if os.path.getsize(RR_protocol_file_name) > 0:  
    with open(RR_protocol_file_name , 'rb') as f :
        RR_protocol_hook = pickle.load(f)


print("average number of successful agents", sum(protocol_hook.num_successful_agents) / len(protocol_hook.num_successful_agents))
print("num episodes", len(protocol_hook.num_successful_agents))

print("average number of successful agents", sum(STR_protocol_hook.num_successful_agents) / len(STR_protocol_hook.num_successful_agents))
print("num episodes", len(STR_protocol_hook.num_successful_agents))

def my_plot_multi_agent_results(ax, plot_this, label = " ", color = 'red', window_size = 10):
    i = 0
    moving_averages = []

    while i < len(plot_this) - window_size + 1:

        window = plot_this[i : i + window_size]

        window_average = round(sum(window) / window_size)

        moving_averages.append(window_average)
        i += 1

    ax.plot(moving_averages, label = label)
    ax.grid()


# protocol_hook.plot()

fig, ax = plt.subplots()
window_size = 500

my_plot_multi_agent_results(ax, protocol_hook.num_successful_agents, label='CSMA-CD', color= 'red' , window_size = window_size)
my_plot_multi_agent_results(ax, STR_protocol_hook.num_successful_agents, label='STR', color= 'green' , window_size = window_size)
my_plot_multi_agent_results(ax, RR_protocol_hook.num_successful_agents, label='RR ', color= 'orange' , window_size = window_size)
my_plot_multi_agent_results(ax, no_protocol_hook.num_successful_agents, label='No Protocol', color= 'blue' , window_size = window_size)

# ax.plot(protocol_hook.num_successful_agents, label = 'protocol')
# ax.plot(no_protocol_hook.num_successful_agents,label = 'no protocol')

ax.set_ylabel('Agents to Complete Task', fontsize=15)
ax.set_xlabel('Episodes', fontsize=15)
plt.title("# Agents Completeing Task Each Training Episode")

plt.legend()
plt.savefig('4_protocols_RR_terrible.png')
plt.show()

