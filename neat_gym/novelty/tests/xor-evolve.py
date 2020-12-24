#!/usr/bin/env python3
'''
Compare Novelty Search with fitness-based search for XOR

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import argparse
import numpy as np
import neat
#from neat_gym.novelty import Novelty
from neat_gym import _NeatConfig, _evolve

def _eval_xor(genome, config):
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitness = 4

    for inp,tgt in zip(((0,0), (0,1), (1,0), (1,1)), (0,1,1,0)):
        out = net.activate(inp)[0]
        fitness -= (out-tgt) ** 2

    return fitness

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--novelty', dest='novelty', action='store_true', help='Use Novelty Search')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Save at each new best')
    parser.add_argument('--ngen', type=int, required=False, help='Number of generations to run')
    parser.add_argument('--seed', type=int, required=False, help='Seed for random number generator')
    args = parser.parse_args()

    np.random.seed(args.seed)

    config = _NeatConfig(
            neat.DefaultGenome, 
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet, 
            neat.DefaultStagnation, 
            'xor-novelty.cfg' if args.novelty else 'xor.cfg', 
            'xor', 
            {'num_inputs':2, 'num_outputs':1}, 
            args.seed)

    _evolve(config, _eval_xor, args.seed, 'xor', args.ngen, args.checkpoint)

if __name__ == '__main__':
    main()
