#!/usr/bin/env python3
'''
(Hyper)NEAT evolver script for OpenAI Gym environemnts

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import multiprocessing as mp
import os
import neat
import argparse
import pickle
import random
from configparser import ConfigParser

from pureples.shared.substrate import Substrate
from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.shared.visualize import draw_net

from neat_gym import eval_net, _GymConfig, _GymHyperConfig

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
    parser.add_argument('--ngen', type=int, required=False, help='Number of generations to run')
    parser.add_argument('--reps', type=int, default=10, required=False, help='Number of repetitions per genome')
    parser.add_argument('--hyper', dest='hyper', action='store_true', help='Use HyperNEAT')
    parser.add_argument('--seed', type=int, required=False, help='Seed for random number generator')
    args = parser.parse_args()

    # Set random seed (including None)
    random.seed(args.seed)

    # Make directories for saving results
    os.makedirs('models', exist_ok=True)
    os.makedirs('visuals', exist_ok=True)

    # If HyperNEAT was requested, use it
    if args.hyper:

        subscfg = ConfigParser()
        subscfg.read(args.env + '.subs')
        coords =  subscfg['Coordinates']
        substrate = Substrate(eval(coords['input']), eval(coords['output']), eval(coords['hidden']))
        actfun = subscfg['Activation']['function']
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
    
    # Add a reporter to save the best
    p.add_reporter(_SaveReporter(args.env))

    # Create a parallel fitness evaluator
    pe = neat.ParallelEvaluator(mp.cpu_count(), evalfun)

    # Run for number of generations specified in config file
    winner_genome = p.run(pe.evaluate) if args.ngen is None else p.run(pe.evaluate, args.ngen) 

    # Save the net to a PDF file

    if args.hyper:
        winner_cppn = neat.nn.FeedForwardNetwork.create(winner_genome, config)
        winner_net = create_phenotype_network(winner_cppn, substrate)
        draw_net(winner_cppn, filename="visuals/winner_cppn")
        draw_net(winner_net, filename="visuals/winner_net")

    else:
        winner_net = neat.nn.FeedForwardNetwork.create(winner_genome, config)
        draw_net(winner_net, filename="visuals/winner_net")

if __name__ == '__main__':

   main()
