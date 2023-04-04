

# Modular RL

## Introduction

This project seeks to solve [Markov Decision Processes (MDP)](https://en.wikipedia.org/wiki/Markov_decision_process) in a modular way. To do so, this project will offer one framework to support classical RL, classical control, multi-agent RL, hierarchical RL, and compositional RL. Doing so allows the end user to 1) understand RL in a modular way and 2) solve complex MDPs that would normally be unsolvable with classical RL approaches.

Please read the following descriptions of each type of RL and how it fits into the framework.



## Classical RL

For the purposes of this project, I am defining classical RL to be typical end-to-end approaches to solving MDPs. These approaches work well for many simple problems but eventually break down as problem complexity increases. Example algorithms are [DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), [DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html), and [PPO](https://arxiv.org/abs/1707.06347). Explaining the details of these algorithms is beyond the scope of this document, but generally speaking they are algorithms designed to solve MDPs via trial and error learning approaches. This means the algorithm tries different actions and uses the results to improve the policy. The policy is just a function which returns an action based on the state of the MDP.

Most if not all classical RL algorithms have the following structure. They consists of only 2 functions: 
* An act function, which takes in the state and returns the action. 
* A learn function, which takes in an experience of interacting with the MDP and updates the policy. Typically, this function call records the experience into a buffer, but only updates the policy every so many steps.

A classical RL algorithm can be represented by the diagram below. The flow from State > Policy > Action represents how states are fed into the policy and an action is returned. The dotted line from Experience > Policy represents how an experience is used to update the policy via whatever algorithm is being used. 


![Internet required to see images](https://github.com/tyler-ingebrand/ModularRL/blob/master/docs/images/Classical%20RL%20Diagram.jpg?raw=true")

I am going to call an object with the above 2 functions (act, learn) an Agent. As we will see, an Agent object can represent many different types of MDP-solving actions from all types of RL and control.




## Control

This framework also supports control. In control, an optimal policy is calculated based on the transition dynamics of a given problem. In the case with a known dynamics function, the control function takes the place of the policy, where it uses the state to calculate an action in some way. Since the transition dynamics are known, the learn function does nothing. If the transition dynamics are unknown, the learn function may update a model of the transition. As a result, the diagram for a control agent looks almost identical to a classical RL agent.

![Internet required to see images](https://github.com/tyler-ingebrand/ModularRL/blob/master/docs/images/Classical%20Control%20Diagram.jpg?raw=true")







## Information Hiding

Often times, some of the information in the state space is redundant and can be removed without making the problem infeasible. Removing redundant information makes the problem easier to solve because the state space is reduced. 

One good example of this is removing symmetries from your MDP. If your MDP has multiple symmetric states with identical actions, you can simply map all of these symmetric states to one state, and then the others do not need to be explored.

There is an equilvalent operation on the output action, where some part of the action space is known and does not need to be learned. A possible example of this is asuming some neutral action for action dimensions that are unused. Assuming an action in this way is less common, but still easily supported.

Both information hiding and action assumptions can be represented by the following diagram. Note the state is transformed by the information hiding function before being passed into the agent. The output action is transformed after before being output. 

The learn function must also apply both of these functions to an experience before passing them to the learn function of the agent.


![Internet required to see images](https://github.com/tyler-ingebrand/ModularRL/blob/master/docs/images/Information%20Hiding%20RL%20Diagram.jpg?raw=true)





## Multi-Agent RL

In multi-agent RL, there are assumed to be multiple decision makers interacting with the same environment. These decision makers may be cooperative, competitive, or some combination thereof. 

In some circumstances, the number of decision makers is constant (such as in a game of chess). In others, the number of decision makers changes (such as cars navigating a highway). 

Sometimes, all agents follow the same policy (all car drivers following the same rules. Due note their actions are still different because they experience a different state). Other times, the agents follow different individual policies (think of 2 robots of different shape. They need different policies because they are not identical). 

Finally, in some situations decision making can be decentralized. This means each agent makes decisions without knowing the actions of the others. In other situations, decision making is centralized. This occurs when success requires interdependent cooperation such that the agents need to know each other's action in order to work togethor. In this case, 1 centralized decision maker may tell each agent what they should do next via an add-on to their state function. 

All of these cases should be representable in a modular framework. Additionally, it would be convenient to reuse classical RL or control solutions. Thus, we end up with the following diagrams:


![Internet required to see images](https://github.com/tyler-ingebrand/ModularRL/blob/master/docs/images/Decentralized%20Multi-Agent%20RL%20Diagram.jpg?raw=true)

To understand this diagram, there are a few notable details.

This diagram shows a decentralized multi-agent scheme, where each agent (marked A sub 1 through A sub N) experiences a local state only with no centralized command structure. 

The global state is split between the agents. For example, assume we have 10 agents, and each agent has a state with 10 values. Thus the global state conists of 100 values. The multi-agent splits this state into 10 agent-specific states. It then feeds the agent-specific states to the respective agents. 

Likewise, the output action is a concatanation of all actions. Assume the same 10 agents have 1 action each. These actions are concatanated to create the output actions. 

Additionally, each agent needs a reward signal specific to its situation. An agent cannot be expected to learn if its reward signal is cluttered with rewards caused by other agents, so we must create a reward function for each agent. Often times, this is one reward function that depends on a agent's local state, and this function is then used for every agent. 

To perform an update, the experience must be divided into agent-specific counterparts, a reward must be calculated for each agent, and then this agent-specific experience must be fed into the update function of the agent.

Notably, this supports having 1 policy for all agents or unique policies for all agents. In the case of 1 policy for all agents, each agent is simply a shallow copy of the original. Thus, in a given update, that 1 agent receives N experience updates. If each policy is unique, then N agents can be created. 

This can also support a variable number of agents, assuming they all follow the same policy. We simply apply that same policy to each division of the state, regardless of how many divisions there are. 

Notably, the agents internally can use classical RL or control algorithms. From the perspective of the Decentralized Multi-Agent, it does not matter which algorithm is used for each agent.


We have a corresponding diagram for centralized multi-agent RL:

![Internet required to see images](https://github.com/tyler-ingebrand/ModularRL/blob/master/docs/images/Centralized%20Multi-Agent%20RL%20Diagram.jpg?raw=true)

This diagram is similiar to before, with a few key differences. First, the splitter becomes a Centralized Command and Splitter. This splits the global state into local states for each agent, and also appends a command to this state telling the agent which task to be doing. Since the centralized commands must also be learned, we must include those commands in the experience. And when we call learn, we must also update the centralized command based on some reward function. 

** I have not worked with centralized multi-agent RL before, so this diagram may need some adjusting. Also, some centralized approaches use a centralized critic but otherwise local decision making?! This should also be possible, but it seems there are many variations of centralized approaches without any clear paradigms yet, at least as far as I know ** 






## Compositional RL

In compositional RL, or CRL, we have multiple agents with the same inputs and outputs. However, their policies are based on different reward functions, and so they produce different behaviors. 

A typical case is break a complex MDP into smaller, easier subtasks, and to train an agent for each subtask. Then, to solve the complex MDP, you simply use the policy of whichever agent should be active given the situation. Often, this can be implemented as a switch statement which decides which agent to use. The corresponding diagram looks like the following:

![Internet required to see images](https://github.com/tyler-ingebrand/ModularRL/blob/master/docs/images/CRL%20Diagram.jpg?raw=true)

**In this example, I only show 3 agents, although you could do any number **

First, you'll notice this looks similiar to a decentralized multi-agent. Unlike in multi-agent RL however, the state is the same for all agents. We use the state as the input to a control function which outputs which agent should be active. Based on that control signal, and a demultiplexor, we send the state to only the 1 agent we want to use. Then, that agent's action is the output. This output is also selected based on the same control signal, although its not shown to avoid cluttering the diagram.

Likewise, an experience update only applies to one of the agents. So, based on the state of the experience, we use our control function to get a control signal. Then, the experience is once again demulitplexed to the correct agent. Only 1 of the agents updates with each call of learn on the Compositional agent. 

As mentioned, the control function can often be a switch statement. I will go over an example in detail later on in this document.



 
## Hierarchical RL

Last but not least, Hierarchical RL (HRL) can also be represented in this framework. HRL is very similiar to CRL except that the control signal is some learned function. This will work better than CRL in cases where it is hard to know which subtasks should be active. 

Additionally, the reward function may be predefined, or it may be based on achieving some goal.

![Internet required to see images](https://github.com/tyler-ingebrand/ModularRL/blob/master/docs/images/HRL%20Diagram.jpg?raw=true)


The act function is identical to before, where the state is sent to a single agent via some control signal. This control signal is based on a learned policy instead of a switch statement however.

The learn function will change somewhat. We still need to Demux the experience and send it to the correct agent. However, since the control function is changing, we want to make sure an experience is sent to the agent that performed it. So, the experience must also contain the control signal. This allows us to make sure the correct agent is updated.

Also, we need to update the learned control based on the experience. Again, it depends on the control signal it sent, which is not typically part of the action, so that will need to be added to the experience as well. We can use this information, and some reward function specific to the control function, to improve the control function. 

In the case that the reward functions for each agent are also changing, that will need to be computed from the experience after demuxing as well.

** HRL is also an area I have little experience. This diagram too may need some changes ** 





## Benefits
### Modularity - Potentially Reuseable
Since agents are modular, it is possible a trained agent could be reused in different tasks. This would save training time as we may only need to learn a policy for a given tasks once, and then learn how to use that policy for other tasks instead of relearning the policy. Please see the example.

### Explainable AI
Given that we have converted agents into modules, we can understand them individually. This allows us to understand a given policy with respect to the reward function we've given it, whereas an end-to-end system may be so complex we cannot do so. Also, especially with CRL, it is much easier to understand overall behavior by simply checking which of the agents is currently active. This provides us some means of explaining behavior produced by the system.

### Problem Complexity Reduction
By breaking up a problem into smaller problems, the overall complexity is reduced. That is, the sum of the complexity of the sub-problems is less than the complexity of the original. This means MDPs that are potentially unsolvable on their own may be feasible if broken down enough times. This may also include information hiding. Please see the example.

### Closed Agent Construction
Since each of the above formulations has the same interface - act, update - the construction of agents is closed. This means we can compose complex agents using other complex agents. I will demonstrate this in the example, but basically this means a compositional agent can make use of other compositional agents, and a hierarchical agent can make use of compositional agents, and so on. Please see the example.

## Example of the Modular Stack

Consider the problem of assembling Ikea furniture using 2 humanoid robots. Clearly, this problem cannot be solved end-to-end. The state space and action space of 2 robots concatanated would be exceptionally large, and the desired behavior equally complex. 

Also, since we have 2 robots, we know this problem could benefit from multi-agent RL (probably centralized). Robots themselves are quite difficult to teach complex behavior, so each robot may benefit from compositional RL. This may even require multiple layers of compositional behavior. Additionally, since not all actions require all of the robots limbs at once, each limb could potentially by considered a different agent. This allows for multi-agent RL to control different limbs and given them different tasks at the same time (consider walking and chewing gum at the same time - Without considering the robot to be composed of multi-agents, this task would have to be learned specifically. Whereas with a multi-agent structure, you would only need to learn walking and chewing independently, and then combine them using multi-agents). At the lowest layer, we will probably need classical RL but also potentially some control algorithms. 

An abbreviated diagram would look something like this:

![Internet required to see images](https://github.com/tyler-ingebrand/ModularRL/blob/master/docs/images/Ikea%20Robots%20Example.jpg?raw=true)

** Arrows are removed for simplicity. This diagram only shows the modules used **
Note how modules are composed of other modules, and some modules can be reused - especially low level tasks such as holding an object, rotating an object, etc. 


