#!/usr/bin/env python3
'''
(Hyper)NEAT evolver script for OpenAI Gym environemnts

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import numpy as np
import multiprocessing as mp
import os
import neat
import argparse
import random

from pureples.shared.substrate import Substrate
from pureples.hyperneat.hyperneat import create_phenotype_network

from neat_gym import eval_net, _GymConfig, _GymHyperConfig, _read_config

class _SaveReporter(neat.reporting.BaseReporter):

    def __init__(self, env_name, checkpoint):

        neat.reporting.BaseReporter.__init__(self)

        self.best = None
        self.env_name = env_name
        self.checkpoint = checkpoint

    def post_evaluate(self, config, population, species, best_genome):

        if self.checkpoint and (self.best is None or best_genome.fitness > self.best):
            self.best = best_genome.fitness
            print('############# Saving new best %f ##############' % self.best)
            config.save_genome(best_genome)

def _eval_genome(genome, config, net, activations):

    fitness = 0

    for _ in range(config.reps):

        fitness += eval_net(net, config.env, activations=activations, seed=config.seed)

    return fitness / config.reps

def _eval_genome_neat(genome, config):

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    return _eval_genome(genome, config, net, 1)

def _eval_genome_hyper(genome, config):

    cppn = neat.nn.FeedForwardNetwork.create(genome, config)
    net = create_phenotype_network(cppn, config.substrate, config.actfun)

    activations = len(config.substrate.hidden_coordinates) + 2

    return _eval_genome(genome, config, net, activations)

def main():

    # Parse command-line arguments

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='Pendulum-v0', help='Environment id')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Save at each new best')
    parser.add_argument('--cfgdir', required=False, default='./config', help='Directory for config files')
    parser.add_argument('--ngen', type=int, required=False, help='Number of generations to run')
    parser.add_argument('--reps', type=int, default=10, required=False, help='Number of repetitions per genome')
    parser.add_argument('--hyper', dest='hyperhid', help='Use HyperNEAT with specified hidden-unit layout')
    parser.add_argument('--seed', type=int, required=False, help='Seed for random number generator')
    args = parser.parse_args()

    # Set random seed (including None)
    random.seed(args.seed)

    # Make directories for saving results
    os.makedirs('models', exist_ok=True)
    os.makedirs('visuals', exist_ok=True)

    # If HyperNEAT was requested, use it
    if args.hyperhid is not None:
        cppncfg = _read_config(args, 'cppn')
        subs =  cppncfg['Substrate']
        nhids = [int(n) for n in args.hyperhid.split(',')]
        hidden = [list(zip(np.linspace(-1,+1,n), [0.]*n)) for n in nhids]
        substrate = Substrate(eval(subs['input']), eval(subs['output']), hidden)
        actfun = subs['function']
        config = _GymHyperConfig(args, substrate, actfun)
        evalfun = _eval_genome_hyper

    # Otherwise, use NEAT
    else:

        config = _GymConfig(args)
        evalfun = _eval_genome_neat

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(show_species_detail=False))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # Add a reporter (which can also checkpoint the best)
    p.add_reporter(_SaveReporter(args.env, args.checkpoint))

    # Create a parallel fitness evaluator
    pe = neat.ParallelEvaluator(mp.cpu_count(), evalfun)

    # Run for number of generations specified in config file
    winner = p.run(pe.evaluate) if args.ngen is None else p.run(pe.evaluate, args.ngen) 

    # Save winner
    config.save_genome(winner)

if __name__ == '__main__':

   main()
