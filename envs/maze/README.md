<img src='media/hard.gif' width=300>

This repository provides an OpenAI Gym environment implementing the maze
challenges in Lehman and Stanely's 2011 Novelty Search
[paper](https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/lehman_ecj11.pdf).
The maze layout and robot update equations are based on [goNEAT_NS](https://github.com/yaricom/goNEAT_NS).

## Quickstart

```
% pip3 install gym
% python3 setup.py install
% python3 gym_nsmaze/envs/medium.py
```

(On Linux you will probably have to run pip3 with sudo.)

You should see the robot taking a [random walk](https://en.wikipedia.org/wiki/Random_walk)
in the medium-difficulty maze.  To see what other options are available with this demo
script, you can do

```
% python3 demos/envs/medium.py --help
```

To run a demo of the more difficult maze, do

```
% python3 demos/envs/hard.py
```

## Running experiments

To replicate the results in Lehman and Stanley's
[paper](https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/lehman_ecj11.pdf), you should first
install [NEAT-Gym](https://github.com/simondlevy/Neat-Gym).  Then do

```
% python3 [DIR]/neat-evolve.py --env gym_nsmaze:Medium-v0 --config config/maze.cfg --novelty --ngen 100
```

where ```[DIR]``` is the directory in which you put NEAT-Gym.

This will run 100 generations of Novelty Search on the medium-difficulty maze using the
[parallel fitness evaluator](https://neat-python.readthedocs.io/en/latest/module_summaries.html#parallel),
so you can take advantage of all the cores on your computer.

Once evolution finishes, you can try out your evolved network by doing:

```
% python3 [DIR]/neat-test.py models/gym_nsmaze:Medium-v0<fitness>.dat
```

where ```<fitness>``` is the fitness of your evolved network. 
