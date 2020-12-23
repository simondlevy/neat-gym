#!/usr/bin/env python3

import argparse
import numpy as np
import neat
from xor import eval_xor
#from neat_gym.novelty import Novelty
from neat_gym import _NeatConfig, _evolve


def xor_fitness(ngen, seed, checkpoint):

    config = _NeatConfig(neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation, 
            'xor.cfg', 'xor', {'num_inputs':3, 'num_outputs':1}, seed)

    np.random.seed(seed)

    _evolve(config, eval_xor, seed, 'xor', ngen, checkpoint)

def xor_novelty(seed=None):

    # Seed the random-number generator for reproducibility.
    np.random.seed(seed)

    # Create an instance of your Novelty class with a k of 10, a threshold of
    # 0.3, and a limit of 150.
    #nov = Novelty(10, 0.3, 150)


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--novelty', dest='novelty', action='store_true', help='Use Novelty Search')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Save at each new best')
    parser.add_argument('--ngen', type=int, required=False, help='Number of generations to run')
    parser.add_argument('--seed', type=int, required=False, help='Seed for random number generator')
    args = parser.parse_args()

    if args.novelty:
        pass

    else:
        xor_fitness(args.ngen, args.seed, args.checkpoint)

if __name__ == '__main__':
    main()
