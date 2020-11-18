#!/usr/bin/env python3
'''
Evolver script for OpenAI Gym environemnts

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import multiprocessing
import os
import visualize
import neat
import pickle
import argparse
import random

from common import eval_genome, GymConfig

def _makedir(name):
    if not os.path.exists(name):
        os.makedirs(name)

def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('environment', metavar='ENVIRONMENT', help='environment (e.g. gym_copter:Lander-v2')
    parser.add_argument('-g', '--ngen', type=int, required=False, help='Number of generations to run')
    parser.add_argument('-r', '--reps', type=int, default=10, required=False, help='Number of repetitions per genome')
    parser.add_argument('-v', '--viz', dest='visualize', action='store_true')
    parser.add_argument('-s', '--seed', type=int, required=False, help='Seed for random number generator')
    args = parser.parse_args()

    # Set random seed (including None)
    random.seed(args.seed)

    # Make directory for pickling nets, if it doesn't already exist
    _makedir(args.environment)

    # Load configuration.
    config = GymConfig(args.environment, args.reps)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Create a parallel fitness evaluator
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)

    # Runn for number of generations specified in config file
    winner = p.run(pe.evaluate) if args.ngen is None else p.run(pe.evaluate, args.ngen) 

    # Pickle the winner 
    filename = '%s/%f.dat' % (args.environment, winner.fitness)
    print('Saving %s' % filename)
    pickle.dump((winner, config), open(filename, 'wb'))

    # Visualize results if indicated
    if args.visualize:
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)

if __name__ == '__main__':

   main()
