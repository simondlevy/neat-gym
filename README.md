# Evolving a neural controller

Like some other [researchers](https://openai.com/blog/evolution-strategies/), I have switched focus recently
from Deep Reinforcement Learning (DRL) to Evolution Strategies (ES).  Current work (with Washington and 
Lee undergraduate student Coletta Fuller) focuses on applying the
[NEAT](https://www.cse.unr.edu/~sushil/class/gas/papers/NEAT.pdf) algorithm to
solve tasks in the GymCopter environment.

To try this out for yourself, you should install [neat-python](https://github.com/CodeReclaimers/neat-python).  Then 
do the following in the gym-copter repository:

```
% cd neat
% python3 evolve.py gym_copter:Lander-v2
```
This will run neat-python using the
[parallel fitness evaluator](https://neat-python.readthedocs.io/en/latest/module_summaries.html#parallel),
so you can take advantage of all the cores on your computer.

Once evolution finishes, you can test out your evolved network by doing:

```
% python3 test.py nets/<fitness>.dat
```

where ```<fitness>``` is the fitness of your evolved network.

To visualize the evolved network you should first install graphviz (```pip3 install graphviz```) and then
run the *show* script:

```
% python3 show.py nets/<fitness>.dat
```
