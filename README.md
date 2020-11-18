This repository contains code allowing you to train, test, and visualize
[OpenAI Gym](https://gym.openai.com/) environments (games) using the
[NEAT](https://www.cse.unr.edu/~sushil/class/gas/papers/NEAT.pdf) algorithm.
The main goal was to make this effort as simple as possible.  For example, the
name of the NEAT configuration file is the same as the name of the environment.

To get started you should install [neat-python](https://github.com/CodeReclaimers/neat-python).  Then 
do the following:

```
% python3 neat-gym-evolve.py LunarLanderContinuous-v2
```
This will run neat-python using the
[parallel fitness evaluator](https://neat-python.readthedocs.io/en/latest/module_summaries.html#parallel),
so you can take advantage of all the cores on your computer.

Once evolution finishes, you can test out your evolved network by doing:

```
% python3 neat-gym-test.py LunarLanderContinuous-v2/<fitness>.dat
```

where ```<fitness>``` is the fitness of your evolved network.

To visualize the evolved network you should first install graphviz (```pip3 install graphviz```) and then
run the *show* script:

```
% python3 neat-gym-show.py LunarLanderContinuous-v2/<fitness>.dat
```

## Similar projects

[neat-openai-gym](https://github.com/sroj/neat-openai-gym)

[OpenAI-NEAT](https://github.com/HackerShackOfficial/OpenAI-NEAT)
