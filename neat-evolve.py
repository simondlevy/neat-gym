#!/usr/bin/env python3
'''
NEAT evolver script for OpenAI Gym environemnts

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from time import time
import os
import argparse
from argparse import ArgumentDefaultsHelpFormatter
import pickle
import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from configparser import ConfigParser

import neat
from neat.math_util import mean, stdev
from neat.reporting import StdOutReporter, BaseReporter
from neat.config import ConfigParameter
from neat.population import Population, CompleteExtinctionException
from neat.nn import FeedForwardNetwork

from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.es_hyperneat.es_hyperneat import ESNetwork
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate

from neat_gym import _gym_make, _is_discrete, eval_net
from neat_gym.novelty import Novelty


class _GymNeatConfig(object):
    '''
    A class for helping Gym work with NEAT
    '''

    __params = [ConfigParameter('pop_size', int),
                ConfigParameter('fitness_criterion', str),
                ConfigParameter('fitness_threshold', float),
                ConfigParameter('reset_on_extinction', bool),
                ConfigParameter('no_fitness_termination', bool, False)]

    def __init__(self, args, layout=None):

        # Check config file exists
        if not os.path.isfile(args.configfile):
            print('No such config file: %s' %
                  os.path.abspath(args.configfile))
            exit(1)

        # Use default NEAT settings
        self.genome_type = neat.DefaultGenome
        self.reproduction_type = neat.DefaultReproduction
        self.species_set_type = neat.DefaultSpeciesSet
        self.stagnation_type = neat.DefaultStagnation

        parameters = ConfigParser()
        with open(args.configfile) as f:
            if hasattr(parameters, 'read_file'):
                parameters.read_file(f)
            else:
                parameters.readfp(f)

            self.node_names = {}

            try:
                names = parameters['Names']
                for idx, name in enumerate(eval(names['input'])):
                    self.node_names[-idx-1] = name
                for idx, name in enumerate(eval(names['output'])):
                    self.node_names[idx] = name
            except Exception:
                pass

        param_list_names = []

        for p in self.__params:
            if p.default is None:
                setattr(self, p.name, p.parse('NEAT', parameters))
            else:
                try:
                    setattr(self, p.name, p.parse('NEAT', parameters))
                except Exception:
                    setattr(self, p.name, p.default)
                    warnings.warn('Using default %s for %s' %
                                  (p.default, p.name), DeprecationWarning)
            param_list_names.append(p.name)

        # Bozo filter for missing sections
        self.check_params(args.configfile, parameters, 'NEAT')
        self.check_params(args.configfile, parameters, 'Gym')

        # Get number of episode repetitions
        gympar = parameters['Gym']
        env_name = gympar['environment']
        self.reps = int(gympar['episode_reps'])

        # Make gym environment form name in command-line arguments
        env = _gym_make(env_name)

        # Get input/output layout from environment, or from layout for Hyper
        if layout is None:
            num_inputs = env.observation_space.shape[0]
            if _is_discrete(env):
                num_outputs = env.action_space.n
            else:
                num_outputs = env.action_space.shape[0]
        else:
            num_inputs, num_outputs = layout

        # Parse type sections.
        genome_dict = dict(parameters.items(self.genome_type.__name__))

        genome_dict['num_inputs'] = num_inputs
        genome_dict['num_outputs'] = num_outputs

        self.genome_config = self.genome_type.parse_config(genome_dict)

        stagnation_dict = dict(parameters.items(self.stagnation_type.__name__))
        self.stagnation_config = \
            self.stagnation_type.parse_config(stagnation_dict)

        self.species_set_dict = \
            dict(parameters.items(self.species_set_type.__name__))
        self.species_set_config = \
            self.species_set_type.parse_config(self.species_set_dict)

        self.reproduction_dict = \
            dict(parameters.items(self.reproduction_type.__name__))
        self.reproduction_config = \
            self.reproduction_type.parse_config(self.reproduction_dict)

        # Store environment name for saving results
        self.env_name = env_name

        # Get number of generations and random seed from config;
        # use defaults if missing
        neatpar = parameters['NEAT']
        self.ngen = self.get_with_default(neatpar, 'generations',
                                          lambda s: int(s), None)
        self.seed = self.get_with_default(neatpar, 'seed',
                                          lambda s: int(s), None)
        self.checkpoint = self.get_with_default(neatpar, 'checkpoint',
                                                lambda s: bool(s), False)

        # Set random seed (including None)
        random.seed(self.seed)

        # Set max episode steps from spec in __init__.py
        self.max_episode_steps = env.spec.max_episode_steps

        # Store environment for later
        self.env = env

        # Track evaluations
        self.current_evaluations = 0
        self.total_evaluations = 0

        # Support novelty search
        self.novelty = _GymNeatConfig.parse_novelty(args.configfile) \
            if args.novelty else None

        # Store config parameters for subclasses
        self.params = parameters

        # For debugging
        self.gen = 0

        # Default to non-recurrent net
        self.activations = 1

    def eval_net_mean(self, net, genome):

        return (self.eval_net_mean_novelty(net, genome)
                if self.is_novelty()
                else self.eval_net_mean_reward(net, genome))

    def eval_net_mean_reward(self, net, genome):

        reward_sum = 0
        total_steps = 0

        for _ in range(self.reps):

            reward, steps = eval_net(net,
                                     self.env,
                                     activations=self.activations,
                                     seed=self.seed,
                                     max_episode_steps=self.max_episode_steps)

            reward_sum += reward
            total_steps += steps

        return reward_sum/self.reps, total_steps

    def eval_net_mean_novelty(self, net, genome):

        reward_sum = 0
        total_steps = 0

        # No behaviors yet
        behaviors = [None] * self.reps

        for j in range(self.reps):

            reward, behavior, steps = self.eval_net_novelty(net, genome)

            reward_sum += reward

            behaviors[j] = behavior

            total_steps += steps

        return reward_sum/self.reps, behaviors, total_steps

    def eval_net_novelty(self, net, genome):

        env = self.env
        env.seed(self.seed)
        state = env.reset()
        steps = 0

        is_discrete = _is_discrete(env)

        total_reward = 0

        while steps < self.max_episode_steps:

            # Support recurrent nets
            for k in range(self.activations):
                action = net.activate(state)

            # Support both discrete and continuous actions
            action = (np.argmax(action)
                      if is_discrete
                      else action * env.action_space.high)

            state, reward, done, info = env.step(action)

            behavior = info['behavior']

            # Accumulate reward, but not novelty
            total_reward += reward

            if done:
                break

            steps += 1

        env.close()

        # Return total reward and final behavior
        return total_reward, behavior, steps

    def save_genome(self, genome):

        name = self.make_name(genome)
        net = FeedForwardNetwork.create(genome, self)
        pickle.dump((net, self.env_name), open('models/%s.dat' % name, 'wb'))
        _GymNeatConfig.draw_net(net,
                                'visuals/%s-network' % name,
                                self.node_names)

    def is_novelty(self):

        return self.novelty is not None

    def make_name(self, genome, suffix=''):

        return '%s%s%+010.3f' % \
               (self.env_name, suffix, genome.actual_fitness)

    def get_with_default(self, params, name, fun, default):
        return fun(params[name]) if name in params else default

    def check_params(self, filename, params, section_name):
        if not params.has_section(section_name):
            self.error('%s section missing from configuration file %s' %
                       (section_name, filename))

    def error(self, msg):
        print('ERROR: ' + msg)
        exit(1)

    @staticmethod
    def draw_net(net, filename, node_names):

        # Create PDF using PUREPLES function
        draw_net(net, filename=filename, node_names=node_names)

        # Delete text
        os.remove(filename)

    @staticmethod
    def eval_genome(genome, config):
        '''
        The result of this function gets assigned to the genome's fitness.
        '''
        net = FeedForwardNetwork.create(genome, config)
        return config.eval_net_mean(net, genome)

    @staticmethod
    def parse_novelty(cfgfilename):

        novelty = None

        parameters = ConfigParser()

        with open(cfgfilename) as f:
            if hasattr(parameters, 'read_file'):
                parameters.read_file(f)
            else:
                parameters.readfp(f)

            try:
                names = parameters['Novelty']
                novelty = Novelty(eval(names['k']),
                                  eval(names['threshold']),
                                  eval(names['limit']),
                                  eval(names['ndims']))
            except Exception:
                print('File %s has no [Novelty] section' % cfgfilename)
                exit(1)

        return novelty


class _GymHyperConfig(_GymNeatConfig):

    def __init__(self, args, substrate=None):

        _GymNeatConfig.__init__(self, args, layout=(5, 1))

        # Attempt to get substrate info from environment
        if hasattr(self.env, 'get_substrate'):
            actfun, inp, hid, out = self.env.get_substrate()

        # Default to substrate info from config file
        else:
            subs = self.params['Substrate']
            inp = eval(subs['input'])
            hid = eval(subs['hidden']) if substrate is None else substrate
            out = eval(subs['output'])
            actfun = subs['function']

        self.substrate = Substrate(inp, out, hid)
        self.actfun = actfun

        # For recurrent nets
        self.activations = len(self.substrate.hidden_coordinates) + 2

        # Output of CPPN is recurrent, so negate indices
        self.node_names = {j: self.node_names[k]
                           for j, k in enumerate(self.node_names)}

        # CPPN itself always has the same input and output nodes
        self.cppn_node_names = {-1: 'x1',
                                -2: 'y1',
                                -3: 'x2',
                                -4: 'y2',
                                -5: 'bias',
                                0: 'weight'}

    def save_genome(self, genome):

        cppn, net = self.make_nets(genome)
        self.save_nets(genome, cppn, net)

    def save_nets(self, genome, cppn, net, suffix='-hyper'):
        pickle.dump((net, self.env_name),
                    open('models/%s.dat' %
                         self.make_name(genome, suffix=suffix), 'wb'))
        _GymNeatConfig.draw_net(cppn,
                                'visuals/%s' %
                                self.make_name(genome, suffix='-cppn'),
                                self.cppn_node_names)
        self.draw_net(net,
                      'visuals/%s' %
                      self.make_name(genome, suffix=suffix),
                      self.node_names)

    def make_nets(self, genome):

        cppn = neat.nn.FeedForwardNetwork.create(genome, self)
        return (cppn,
                create_phenotype_network(cppn,
                                         self.substrate,
                                         self.actfun))

    @staticmethod
    def eval_genome(genome, config):

        cppn, net = config.make_nets(genome)
        return config.eval_net_mean(net, genome)


class _GymEsHyperConfig(_GymHyperConfig):

    def __init__(self, args):

        _GymHyperConfig.__init__(self, args, substrate=())

        es = self.params['ES']

        self.es_params = {
                'initial_depth': int(es['initial_depth']),
                'max_depth': int(es['max_depth']),
                'variance_threshold': float(es['variance_threshold']),
                'band_threshold': float(es['band_threshold']),
                'iteration_level': int(es['iteration_level']),
                'division_threshold': float(es['division_threshold']),
                'max_weight': float(es['max_weight']),
                'activation': es['activation']
                }

    def save_genome(self, genome):

        cppn, _, net = self.make_nets(genome)
        self.save_nets(genome, cppn, net, suffix='-eshyper')

    def make_nets(self, genome):

        cppn = neat.nn.FeedForwardNetwork.create(genome, self)
        esnet = ESNetwork(self.substrate, cppn, self.es_params)
        net = esnet.create_phenotype_network()
        return cppn, esnet, net

    @staticmethod
    def eval_genome(genome, config):

        _, esnet, net = config.make_nets(genome)
        return config.eval_net_mean(net, genome)


class _GymPopulation(Population):
    '''
    Supports genomes that report their number of evaluations
    '''

    def __init__(self, config, stats):

        Population.__init__(self, config)

        self.config = config

        self.stats = stats

    def run(self, fitness_function, ngen, maxtime):

        gen = 0
        start = time()

        while ((ngen is None or gen < ngen)
               and (maxtime is None or time()-start < maxtime)):

            self.config.gen = gen

            gen += 1

            self.config.current_evaluations = 0

            self.reporters.start_generation(self.generation)

            # Evaluate all genomes using the user-provided function.
            fitness_function(list(self.population.items()), self.config)

            # Gather and report statistics.
            best = None
            for g in self.population.values():

                if g.fitness is None:
                    raise RuntimeError('Fitness not assigned to genome %d' %
                                       g.key)

                # Break out fitness tuple into actual fitness, evaluations
                g.fitness, g.actual_fitness, evaluations = (
                        self.parse_fitness(g.fitness))

                # Accumulate evaluations
                self.config.current_evaluations += evaluations
                self.config.total_evaluations += evaluations

                if best is None:
                    best = g

                else:
                    if g.actual_fitness > best.actual_fitness:
                        best = g

            self.reporters.post_evaluate(self.config,
                                         self.population,
                                         self.species,
                                         best)

            # Track the best genome ever seen.
            if (self.best_genome is None or
                    best.actual_fitness > self.best_genome.actual_fitness):
                self.best_genome = best

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.actual_fitness
                                            for g in self.population.values())
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config,
                                                  self.generation,
                                                  best)
                    break

            # Create the next generation from the current generation.
            self.reproduce()

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.create_new_pop()
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config,
                                  self.population,
                                  self.generation)

            self.reporters.end_generation(self.config,
                                          self.population,
                                          self.species)

            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config,
                                          self.generation,
                                          self.best_genome)

        self.plot_species()

        return self.best_genome

    def reproduce(self):
        self.population = \
                 self.reproduction.reproduce(self.config, self.species,
                                             self.config.pop_size,
                                             self.generation)

    def create_new_pop(self):
        self.population = \
                self.reproduction.create_new(self.config.genome_type,
                                             self.config.genome_config,
                                             self.config.pop_size)

    def parse_fitness(self, fitness):
        '''
        Break out fitness tuple into
        (fitness for selection, actual fitness, evaluations)
        '''
        return fitness[0], fitness[0], fitness[1]

    def plot_species(self):
        """ Visualizes speciation throughout evolution. """

        species_sizes = self.stats.get_species_sizes()
        num_generations = len(species_sizes)
        curves = np.array(species_sizes).T

        fig, ax = plt.subplots()
        ax.stackplot(range(num_generations), *curves)

        filename = self.config.make_name(self.best_genome)

        plt.title(filename)
        plt.ylabel("Size per Species")
        plt.xlabel("Generations")

        plt.savefig('visuals/%s-species.pdf' % filename)

        plt.close()


class _NoveltyPopulation(_GymPopulation):
    '''
    Supports genomes that report their novelty
    '''

    def __init__(self, config, stats):

        _GymPopulation.__init__(self, config, stats)

    def parse_fitness(self, fitness):
        '''
        Break out fitness tuple into
        (fitness for selection, actual fitness, evaluations)
        '''

        # Use actual_fitness to encode ignored objective, and replace genome's
        # fitness with its novelty, summed over behaviors.  If the behavior is
        # None, we treat its sparsity as zero.
        actual_fitness, behaviors, evaluations = fitness

        fitness = np.sum([0 if behavior is None
                          else self.config.novelty.add(behavior)
                          for behavior in behaviors])

        return fitness, actual_fitness, evaluations


class _SaveReporter(BaseReporter):

    def __init__(self, env_name, checkpoint, novelty):

        BaseReporter.__init__(self)

        self.best_fitness = -np.inf
        self.checkpoint = checkpoint

        # Make directories for results
        _SaveReporter.mkdir('models')
        _SaveReporter.mkdir('visuals')
        _SaveReporter.mkdir('runs')

        # Create CSV file for history and write its header
        self.csvfile = open('runs/%s.csv' % env_name, 'w')
        self.csvfile.write('Gen,Time,MeanFit,StdFit,MaxFit')
        if novelty:
            self.csvfile.write(',MeanNov,StdNov,MaxNov')
        self.csvfile.write('\n')

        # Start timing for CSV file data
        self.start = time()

    def post_evaluate(self, config, population, species, best_genome):

        fits = [c.actual_fitness for c in population.values()]

        # Save current generation info to history file
        fit_max = max(fits)
        self.csvfile.write('%d,%f,%+5.3f,%+5.3f,%+5.3f' %
                           (config.gen,
                            time()-self.start,
                            mean(fits),
                            stdev(fits),
                            fit_max))

        if config.is_novelty():
            novs = [c.fitness for c in population.values()]
            self.csvfile.write(',%+5.3f,%+5.3f,%+5.3f' %
                               (mean(novs), stdev(novs), max(novs)))

        self.csvfile.write('\n')
        self.csvfile.flush()

        # Track best
        if self.checkpoint and fit_max > self.best_fitness:
            self.best_fitness = fit_max
            print('############# Saving new best %f ##############' %
                  self.best_fitness)
            config.save_genome(best_genome)

    def mkdir(name):
        os.makedirs(name, exist_ok=True)


class _StdOutReporter(StdOutReporter):

    def __init__(self, show_species_detail):

        StdOutReporter.__init__(self, show_species_detail)

    def post_evaluate(self, config, population, species, best_genome):

        # Special report for novelty search
        if config.is_novelty():

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

        # Ordinary report otherwise
        else:

            StdOutReporter.post_evaluate(
                    self,
                    config,
                    population,
                    species,
                    best_genome)

        print('Evaluations this generation: %d' % config.current_evaluations)
        print('Total evaluations: %d' % config.total_evaluations)


def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('configfile', metavar='CONFIGFILE',
                        help='input config file')
    group.add_argument('--hyper', action='store_true', help='Use HyperNEAT')
    group.add_argument('--eshyper', action='store_true',
                       help='Use ES-HyperNEAT')
    parser.add_argument('--novelty', action='store_true',
                        help='Use Novelty Search')
    parser.add_argument('--maxtime', default=None, type=int,
                        help='Maximum time in seconds')
    args = parser.parse_args()

    # Check for HyperNEAT, ES-HyperNEAT
    if args.hyper:
        config = _GymHyperConfig(args)
    if args.eshyper:
        config = _GymEsHyperConfig(args)
    # Default to original NEAT
    else:
        config = _GymNeatConfig(args)

    # Create a statistics reporter
    stats = neat.StatisticsReporter()

    # Create an ordinary population or a population for NoveltySearch
    pop = (_NoveltyPopulation(config, stats)
           if config.is_novelty()
           else _GymPopulation(config, stats))

    # Add a stdout reporter to show progress in the terminal
    pop.add_reporter(_StdOutReporter(show_species_detail=False))
    pop.add_reporter(stats)

    # Add a reporter (which can also checkpoint the best)
    pop.add_reporter(_SaveReporter(config.env_name,
                                   config.checkpoint,
                                   args.novelty))

    # Create a parallel fitness evaluator
    pe = neat.ParallelEvaluator(mp.cpu_count(), config.eval_genome)

    # Run for number of generations specified in config file
    winner = pop.run(pe.evaluate, config.ngen, args.maxtime)

    # Save winner
    config.save_genome(winner)


if __name__ == '__main__':
    main()
