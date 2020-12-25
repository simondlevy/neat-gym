#!/usr/bin/env python3
'''
Compare Novelty Search with fitness-based search for XOR

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import argparse
import numpy as np
import neat
from neat_gym import NoveltyGenome, NeatConfig, evolve

def _eval_xor_both(genome, config):
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitness = 4
    outputs = [0]*4

    for k,inp,tgt in zip(range(4), ((0,0), (0,1), (1,0), (1,1)), (0,1,1,0)):
        out = net.activate(inp)[0]
        outputs[k] = out
        fitness -= (out-tgt) ** 2
    
    return outputs, fitness

def _eval_xor_fitness(genome, config):

    return _eval_xor_both(genome, config)[1]

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--novelty', dest='novelty', action='store_true', help='Use Novelty Search')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Save at each new best')
    parser.add_argument('--ngen', type=int, required=False, help='Number of generations to run')
    parser.add_argument('--seed', type=int, required=False, help='Seed for random number generator')
    args = parser.parse_args()

    np.random.seed(args.seed)

    config = NeatConfig(
            NoveltyGenome if args.novelty else neat.DefaultGenome, 
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet, 
            neat.DefaultStagnation, 
            'xor-novelty.cfg' if args.novelty else 'xor.cfg', 
            'xor', 
            {'num_inputs':2, 'num_outputs':1}, 
            args.seed)

    evalfun = _eval_xor_both if args.novelty else _eval_xor_fitness

    evolve(config, evalfun, args.seed, 'xor', args.ngen, args.checkpoint)

if __name__ == '__main__':
    main()
