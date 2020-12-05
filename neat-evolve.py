#!/usr/bin/env python3
'''
NEAT evolver script for OpenAI Gym environemnts

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import multiprocessing
import os
import neat
import argparse
import pickle
import random
from neat_gym import visualize, eval_net, _GymConfig

class _SaveReporter(neat.reporting.BaseReporter):

    def __init__(self, env_name):

        neat.reporting.BaseReporter.__init__(self)

        self.best = None
        self.env_name = env_name

    def post_evaluate(self, config, population, species, best_genome):

        if self.best is None or best_genome.fitness > self.best:
            self.best = best_genome.fitness
            filename = 'models/%s%f.dat' % (self.env_name, best_genome.fitness)
            print('Saving %s' % filename)
            pickle.dump((best_genome, config), open(filename, 'wb'))

def _eval_genome(genome, config):

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitness = 0

    for _ in range(config.reps):

        fitness += eval_net(net, config.env)

    return fitness / config.reps

def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='Pendulum-v0', help='Environment id')
    parser.add_argument('--ngen', type=int, required=False, help='Number of generations to run')
    parser.add_argument('--reps', type=int, default=10, required=False, help='Number of repetitions per genome')
    parser.add_argument('--viz', dest='visualize', action='store_true', help='Visualize evolution history')
    parser.add_argument('--seed', type=int, required=False, help='Seed for random number generator')
    args = parser.parse_args()

    # Set random seed (including None)
    random.seed(args.seed)

    # Make directory for pickling nets
    os.makedirs('models', exist_ok=True)

    # Load configuration.
    config = _GymConfig(args.env, args.reps)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(show_species_detail=False))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # Add a reporter to save the best
    p.add_reporter(_SaveReporter(args.env))

    # Create a parallel fitness evaluator
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), _eval_genome)

    # Run for number of generations specified in config file
    p.run(pe.evaluate) if args.ngen is None else p.run(pe.evaluate, args.ngen) 

    # Visualize results if indicated
    if args.visualize:
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)

if __name__ == '__main__':

   main()
