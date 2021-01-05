#!/usr/bin/env python3
'''
NEAT evolver script for OpenAI Gym environemnts

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import argparse
from argparse import ArgumentDefaultsHelpFormatter
import random
import os
import multiprocessing as mp
import numpy as np
from neat.math_util import mean, stdev
import neat
from neat.reporting import StdOutReporter, BaseReporter
import neat_gym


class _SaveReporter(BaseReporter):

    def __init__(self, env_name, checkpoint):

        BaseReporter.__init__(self)

        self.best_fitness = -np.inf
        self.env_name = env_name
        self.checkpoint = checkpoint

    def post_evaluate(self, config, population, species, best_genome):

        best_genome_fitness = config.get_actual_fitness(best_genome)

        if self.checkpoint and best_genome_fitness > self.best_fitness:
            self.best_fitness = best_genome_fitness
            print('############# Saving new best %f ##############' %
                  self.best_fitness)
            config.save_genome(best_genome)


class _StdOutReporter(StdOutReporter):

    def __init__(self, show_species_detail):

        StdOutReporter.__init__(self, show_species_detail)

    def post_evaluate(self, config, population, species, best_genome):
        if config.novelty is None:
            StdOutReporter.post_evaluate(
                    self,
                    config,
                    population,
                    species,
                    best_genome)
            return
        novelties = [c.fitness for c in population.values()]
        nov_mean = mean(novelties)
        nov_std = stdev(novelties)
        best_species_id = species.get_species_id(best_genome.key)
        print('Population\'s average novelty: %3.5f stdev: %3.5f' %
              (nov_mean, nov_std))
        print('Best novelty: %3.5f - size: (%d,%d) - species %d - id %d' %
              (best_genome.fitness,
               best_genome.size()[0],
               best_genome.size()[1],
               best_species_id,
               best_genome.key))
        print('Best actual fitness: %f ' % best_genome.actual_fitness)


def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--hyper', action='store_true', help='Use HyperNEAT')
    group.add_argument('--eshyper', action='store_true',
                       help='Use ES-HyperNEAT')
    parser.add_argument('--novelty', action='store_true',
                        help='Use Novelty Search')
    parser.add_argument('--env', dest='env_name', default='CartPole-v1',
                        help='Environment name')
    parser.add_argument('--checkpoint', action='store_true',
                        help='Save at each new best')
    parser.add_argument('--config', required=False, default=None,
                        help='Config file; if None, config/<env-name>.cfg')
    parser.add_argument('--ngen', type=int, required=False,
                        help='Number of generations to run')
    parser.add_argument('--reps', type=int, default=10, required=False,
                        help='Number of episode repetitions per genome')
    parser.add_argument('--seed', type=int, required=False,
                        help='Seed for random number generator')
    args = parser.parse_args()

    # Default to original NEAT
    config = neat_gym._GymNeatConfig(args)

    # Check for HyperNEAT, ES-HyperNEAT
    if args.hyper:
        config = neat_gym._GymHyperConfig(args)
    if args.eshyper:
        config = neat_gym._GymEsHyperConfig(args)

    # Set random seed (including None)
    random.seed(args.seed)

    # Make directories for saving results
    os.makedirs('models', exist_ok=True)
    os.makedirs('visuals', exist_ok=True)

    # Create an ordinary population or a population for NoveltySearch
    pop = (neat_gym._NoveltyPopulation(config)
           if config.is_novelty()
           else neat.Population(config))

    # Add a stdout reporter to show progress in the terminal.
    pop.add_reporter(_StdOutReporter(show_species_detail=False))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Add a reporter (which can also checkpoint the best)
    pop.add_reporter(_SaveReporter(args.env_name, args.checkpoint))

    # Create a parallel fitness evaluator
    pe = neat.ParallelEvaluator(mp.cpu_count(), config.eval_genome)

    # Run for number of generations specified in config file
    winner = (pop.run(pe.evaluate)
              if args.ngen is None
              else pop.run(pe.evaluate, args.ngen))

    # Save winner
    config.save_genome(winner)


if __name__ == '__main__':
    main()
