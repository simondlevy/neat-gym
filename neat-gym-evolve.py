#!/usr/bin/env python3
'''
Evolver script for OpenAI Gym environemnts

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import multiprocessing
import os
import neat
import pickle
import argparse
import random
from neat_gym import visualize

from neat_gym import eval_net, _GymConfig

def _eval_genome(genome, config):

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitness = 0

    for _ in range(config.reps):

        fitness += eval_net(net, config.env)

    return fitness / config.reps

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

    # Make directory for pickling nets
    os.makedirs(args.environment, exist_ok=True)

    # Load configuration.
    config = _GymConfig(args.environment, args.reps)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Create a parallel fitness evaluator
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), _eval_genome)

    # Run for number of generations specified in config file
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
