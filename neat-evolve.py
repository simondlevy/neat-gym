#!/usr/bin/env python3
'''
(Hyper)NEAT evolver script for OpenAI Gym environemnts

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import numpy as np
import multiprocessing as mp
import os
import argparse
import random

import neat

from pureples.shared.substrate import Substrate

from neat_gym import _GymConfig, _GymHyperConfig, _GymEsHyperConfig
from neat_gym import _eval_genome_neat, _eval_genome_hyper, _eval_genome_eshyper

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


def main():

    # Parse command-line arguments

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='Pendulum-v0', help='Environment id')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Save at each new best')
    parser.add_argument('--cfgdir', required=False, default='./config', help='Directory for config files')
    parser.add_argument('--ngen', type=int, required=False, help='Number of generations to run')
    parser.add_argument('--reps', type=int, default=10, required=False, help='Number of repetitions per genome')
    parser.add_argument('--hyper', dest='hyperhid', help='Use HyperNEAT with specified hidden-unit layout or ES')
    parser.add_argument('--seed', type=int, required=False, help='Seed for random number generator')
    args = parser.parse_args()

    # Set random seed (including None)
    random.seed(args.seed)

    # Make directories for saving results
    os.makedirs('models', exist_ok=True)
    os.makedirs('visuals', exist_ok=True)

    # If HyperNEAT was requested, use it
    if args.hyperhid is not None:

        cfg = _GymConfig.load(args, '-eshyper' if args.hyperhid == 'es' else '-hyper')
        subs =  cfg['Substrate']
        actfun = subs['function']
        inp = eval(subs['input'])
        out = eval(subs['output'])

        if args.hyperhid == 'es':

            substrate = Substrate(inp, out)
            cfg = _GymEsHyperConfig(args, substrate, actfun)
            exit(0)

        else:

            cfg = _GymConfig.load(args, '-hyper')

            try:
                nhids = [int(n) for n in args.hyperhid.split(',')]
            except:
                print('Hidden-unit layout should be a number or tuple of numbers')
                exit(1)

            hid = [list(zip(np.linspace(-1,+1,n), [0.]*n)) for n in nhids]
            evalfun = _eval_genome_hyper
            substrate = Substrate(inp, out, hid)
            config = _GymHyperConfig(args, substrate, actfun)

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
