#!/usr/bin/env python3

import argparse
import numpy as np
import neat
#from neat_gym.novelty import Novelty

def _eval_xor(genome, config):
    '''
    Must be global for pickling.
    '''
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    sse = 0

    for inp,tgt in zip(((0,0), (0,1), (1,0), (1,1)), (0,1,1,0)):
        sse += (tgt - net.activate(inp + (1,))[0])**2

    return 1 - np.sqrt(sse/4)

def xor_fitness(ngen=1000, seed=None, checkpoint=True):

    import neat
    from neat_gym import _NeatConfig, _evolve

    config = _NeatConfig(neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation, 
            'xor.cfg', 'xor', {'num_inputs':3, 'num_outputs':1}, seed)

    np.random.seed(seed)

    _evolve(config, _eval_xor, seed, 'xor', ngen, checkpoint)

def xor_novelty(seed=None):

    # Seed the random-number generator for reproducibility.
    np.random.seed(seed)

    # Create an instance of your Novelty class with a k of 10, a threshold of
    # 0.3, and a limit of 150.
    #nov = Novelty(10, 0.3, 150)


def main():

    tasks = ['fitness', 'novelty']

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task', required=True, help='(' + ', '.join(tasks) + ')')
    parser.add_argument('--ngen', type=int, required=False, help='Number of generations to run')
    parser.add_argument('--seed', type=int, required=False, help='Seed for random number generator')
    args = parser.parse_args()

    if args.task == 'fitness':
        xor_fitness()
    elif args.task == 'novelty':
        pass
    else:
        print('Task must be one of: ' + ', '.join(tasks))
        exit(0)

if __name__ == '__main__':
    main()
