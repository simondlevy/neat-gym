#!/usr/bin/env python3
'''
(Hyper)NEAT evolver script for OpenAI Gym environemnts

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import multiprocessing as mp
import os
import argparse
import random

import neat

from pureples.shared.substrate import Substrate

from neat_gym import _GymConfig, _GymEsHyperConfig, _SaveReporter

def main():

    # Parse command-line arguments

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='CartPole-v1', help='Environment id')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Save at each new best')
    parser.add_argument('--cfgdir', required=False, default='./config', help='Directory for config files')
    parser.add_argument('--ngen', type=int, required=False, help='Number of generations to run')
    parser.add_argument('--reps', type=int, default=10, required=False, help='Number of repetitions per genome')
    parser.add_argument('--seed', type=int, required=False, help='Seed for random number generator')
    args = parser.parse_args()

    # Set random seed (including None)
    random.seed(args.seed)

    # Make directories for saving results
    os.makedirs('models', exist_ok=True)
    os.makedirs('visuals', exist_ok=True)

    cfg = _GymConfig.load(args, '-eshyper')
    subs =  cfg['Substrate']
    actfun = subs['function']
    inp = eval(subs['input'])
    out = eval(subs['output'])

    evalfun = _GymEsHyperConfig.eval_genome
    substrate = Substrate(inp, out)
    config = _GymEsHyperConfig(args, substrate, actfun, cfg['ES'])

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

main()
