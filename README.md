This repository contains code allowing you to train, test, and visualize
[OpenAI Gym](https://gym.openai.com/) environments (games) using the
[NEAT](https://www.cse.unr.edu/~sushil/class/gas/papers/NEAT.pdf) algorithm.

The two goals of this project are 

1. Make this work as simple as possible.  For example, the name of the NEAT
configuration file defaults to the name of the environment.

2. Make the code run fast, by simultaneously evaluating the fitnesses of the
population on multiprocessor machines.

To get started you should install [neat-python](https://github.com/CodeReclaimers/neat-python) 
and [PUREPLES](https://github.com/ukuleleplayer/pureples) from source. Then 
do the following:

```
% python3 neat-evolve.py 
```
This will run neat-python on the default [CartPole-v1](https://gym.openai.com/envs/CartPole-v1/) environment using the
[parallel fitness evaluator](https://neat-python.readthedocs.io/en/latest/module_summaries.html#parallel),
so you can take advantage of all the cores on your computer.

Once evolution finishes, you can try out your evolved network by doing:

```
% python3 neat-test.py models/CartPole-v1<fitness>.dat
```

where ```<fitness>``` is the fitness of your evolved network.

## HyperNEAT and ES-HyperNEAT

NEAT-Gym supports [HyperNEAT](https://en.wikipedia.org/wiki/HyperNEAT) via the ```--hyper``` option and
and [ES-HyperNEAT](http://eplex.cs.ucf.edu/ESHyperNEAT/) via the <br> ```--eshyper``` option.

## Novelty Search

NEAT-Gym supports
[Novelty Search](https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/lehman_ecj11.pdf)
via the ```--novelty``` option.  To use this option, your environment should provide a
```step_novelty``` method.  Like the ordinary ```step``` method, ```step_novelty``` should 
accept an action as input; however, instead of returning a tuple ```state, reward, done, info```
it should return ```state, reward, behavior, done, info```, where ```behavior``` is the behavior
of the agent at the end of the episode (for example, its final position in the
maze), or ```None``` before the end of the episode.  

[gym-nsmaze](https://github.com/simondlevy/gym-nsmaze)
provides this capability, using the medium/hard maze environments in the Novelty Search
[paper](https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/lehman_ecj11.pdf).

## Using NEAT-Gym in other projects

The
[neat_gym](https://github.com/simondlevy/NEAT-Gym/blob/master/neat_gym/__init__.py)
module has functions for loading a .dat file and using it to run an episode of
your Gym environment.  To make these functions available other projects (e.g.,
3D animation of your environment), do:

```
% sudo python3 setup.py install
```

You will also need to create a NEAT configuration file for your environment.  As usual,
the easiest way to do this is to take something that works (like the config file for
[CartPole-v1](https://github.com/simondlevy/neat-gym/blob/master/config/CartPole-v1.cfg)
and modify it to do what you want.

## Related projects

* [neat-openai-gym](https://github.com/sroj/neat-openai-gym)

* [OpenAI-NEAT](https://github.com/HackerShackOfficial/OpenAI-NEAT)

* [AC-Gym](https://github.com/simondlevy/AC-Gym)

* [CMA-Gym](https://github.com/simondlevy/CMA-Gym)
