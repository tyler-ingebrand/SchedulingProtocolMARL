# Decentralized Conflict Resolution for Multi-Agent Reinforcement Learning Through Shared Scheduling Protocols
### By Tyler Ingebrand*, Sophia Smith*, and Ufuk Topcu.

This is the official repo for "Decentralized Conflict Resolution for Multi-Agent Reinforcement Learning Through Shared Scheduling Protocols" submitted to CDC. Experimental results are included in gridgame/results, and can be reproduced using run_experiments.py. 

Gridgame is a gridworld with a variable number of agents acting the hallway environment. It follows the [Petting Zoo interface](https://github.com/Farama-Foundation/PettingZoo). MARL is handled by a seperate repo, [added as a submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules), called [ModularRL](https://github.com/tyler-ingebrand/ModularRL). ModularRL allows for the easy creation of MARL learners via a modular interface. 


<img src=https://github.com/tyler-ingebrand/SchedulingProtocolMARL/blob/main/gridgame/screenshot.jpg data-canonical-src=https://github.com/tyler-ingebrand/SchedulingProtocolMARL/blob/main/gridgame/screenshot.jpg width="400" height="400" />

### Installation
The major packages required are numpy, matplotlib, PettingZoo, gym, and gynmnasium, which will automatically install many of the packages below. 
Keep in mind you must also install the submodule, register it with "pip install -e ModularRL", and also install its dependencies such as torch and sb3. 
The python version used is 3.9.13. 

The following versions are reported by pip:
- cloudpickle==2.2.1
- colorama==0.4.6
- contourpy==1.0.7
- cycler==0.11.0
- fonttools==4.39.0
- gym==0.21.0
- gym-notices==0.0.8
- gymnasium==0.27.1
- gymnasium-notices==0.0.1
- importlib-metadata==4.13.0
- importlib-resources==5.12.0
- jax-jumpy==0.2.0
- kiwisolver==1.4.4
- matplotlib==3.7.1
- -e git+https://github.com/tyler-ingebrand/ModularRL.git@691bf735bd7f5e2a098a3a74bcce8b61a5bd61ac#egg=ModularRL&subdirectory=..\..\..\gridgame\modularrl
- numpy==1.24.2
- packaging==23.0
- pandas==1.5.3
- PettingZoo==1.22.3
- Pillow==9.4.0
- pygame==2.2.0
- pyparsing==3.0.9
- python-dateutil==2.8.2
- pytz==2022.7.1
- six==1.16.0
- stable-baselines3==1.7.0
- torch==1.13.1
- tqdm==4.65.0
- typing_extensions==4.5.0
- zipp==3.15.0
