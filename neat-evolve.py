#!/usr/bin/env python3
'''
NEAT evolver script for OpenAI Gym environemnts

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import os
import argparse
from argparse import ArgumentDefaultsHelpFormatter
import pickle
import warnings
import random
import numpy as np
import multiprocessing as mp
from configparser import ConfigParser

import neat
from neat.math_util import mean, stdev
from neat.reporting import StdOutReporter, BaseReporter

from neat.config import ConfigParameter, UnknownConfigItemError
from neat.population import Population, CompleteExtinctionException
from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.es_hyperneat.es_hyperneat import ESNetwork
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate

from neat_gym import _gym_make, _is_discrete, _eval_net
from neat_gym.novelty import Novelty


def _parse_novelty(cfgfilename):

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


class NeatConfig(object):
    '''
    Replaces neat.Config to support Novelty Search.
    '''

    __params = [ConfigParameter('pop_size', int),
                ConfigParameter('fitness_criterion', str),
                ConfigParameter('fitness_threshold', float),
                ConfigParameter('reset_on_extinction', bool),
                ConfigParameter('no_fitness_termination', bool, False)]

    def __init__(self,
                 genome_type,
                 reproduction_type,
                 species_set_type,
                 stagnation_type,
                 config_file_name,
                 env_name,
                 layout_dict,
                 seed,
                 novelty=False):

        # Check that the provided types have the required methods.
        assert hasattr(genome_type, 'parse_config')
        assert hasattr(reproduction_type, 'parse_config')
        assert hasattr(species_set_type, 'parse_config')
        assert hasattr(stagnation_type, 'parse_config')

        self.genome_type = genome_type
        self.reproduction_type = reproduction_type
        self.species_set_type = species_set_type
        self.stagnation_type = stagnation_type
        self.env_name = env_name
        self.seed = seed

        if not os.path.isfile(config_file_name):
            raise Exception('No such config file: %s' %
                            os.path.abspath(config_file_name))

        parameters = ConfigParser()
        with open(config_file_name) as f:
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

            # For recurrent nets (default to feed-forward)
            self.activations = 1
            genome_params = parameters['DefaultGenome']
            try:
                self.activations = int(genome_params['activations'])
            except Exception:
                pass

        # NEAT configuration
        if not parameters.has_section('NEAT'):
            raise RuntimeError('NEAT section missing from configuration file.')

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

        param_dict = dict(parameters.items('NEAT'))
        unknown_list = [x for x in param_dict if x not in param_list_names]
        if unknown_list:
            if len(unknown_list) > 1:
                raise UnknownConfigItemError(
                        'Unknown (section NEAT) configuration items:\n' +
                        '\n\t'.join(unknown_list))
            raise UnknownConfigItemError(
                'Unknown (section NEAT) configuration item %s' %
                format(unknown_list[0]))

        # Parse type sections.
        genome_dict = dict(parameters.items(genome_type.__name__))

        # Add layout (input/output) info
        for key in layout_dict:
            genome_dict[key] = layout_dict[key]

        self.genome_config = genome_type.parse_config(genome_dict)

        species_set_dict = dict(parameters.items(species_set_type.__name__))
        self.species_set_config = \
            species_set_type.parse_config(species_set_dict)

        stagnation_dict = dict(parameters.items(stagnation_type.__name__))
        self.stagnation_config = stagnation_type.parse_config(stagnation_dict)

        reproduction_dict = dict(parameters.items(reproduction_type.__name__))
        self.reproduction_config = \
            reproduction_type.parse_config(reproduction_dict)

        # Support novelty search
        self.novelty = _parse_novelty(config_file_name) if novelty else None

        # Store config parameters for subclasses
        self.params = parameters

    def save_genome(self, genome):

        name = self.make_name(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, self)
        pickle.dump((net, self.env_name), open('models/%s.dat' % name, 'wb'))
        _GymNeatConfig.draw_net(net, 'visuals/%s' % name, self.node_names)

    def is_novelty(self):

        return self.novelty is not None

    def get_actual_fitness(self, genome):

        return genome.actual_fitness if self.is_novelty() else genome.fitness

    def make_name(self, genome, suffix=''):

        return '%s%s%+010.3f' % \
               (self.env_name, suffix, self.get_actual_fitness(genome))


class _GymNeatConfig(NeatConfig):
    '''
    A class for helping Gym work with NEAT
    '''

    def __init__(self, args, layout=None):

        # Make gym environment form name in command-line arguments
        env = _gym_make(args.env_name)

        # Make sure environment supports novelty
        if args.novelty:
            unenv = env.unwrapped
            if not hasattr(unenv, 'step_novelty'):
                print('Error: environment %s does not support novelty search' %
                      args.env_name)
                exit(1)

        # Get input/output layout from environment, or from layout for Hyper
        if layout is None:
            num_inputs = env.observation_space.shape[0]
            if _is_discrete(env):
                num_outputs = env.action_space.n
            else:
                num_outputs = env.action_space.shape[0]
        else:
            num_inputs, num_outputs = layout

        # Default to environment name for config file
        cfgfilename = ('config/' + args.env_name + '.cfg'
                       if args.config is None else args.config)

        # Do non-Gym config stuff
        NeatConfig.__init__(self,
                            neat.DefaultGenome,
                            neat.DefaultReproduction,
                            neat.DefaultSpeciesSet,
                            neat.DefaultStagnation,
                            cfgfilename,
                            args.env_name,
                            {'num_inputs': num_inputs,
                             'num_outputs': num_outputs},
                            args.seed,
                            args.novelty)

        # Get number of episode repetitions
        gympar = self.params['Gym']
        self.reps = int(gympar['episode_reps'])

        # Store environment for later
        self.env = env

        # Track evaluations
        self.current_evaluations = 0
        self.total_evaluations = 0

    def eval_net_mean(self, net, activations):

        return (self.eval_net_mean_novelty(net, activations)
                if self.is_novelty()
                else self.eval_net_mean_reward(net, activations))

    def eval_net_mean_reward(self, net, activations):

        reward_sum = 0
        total_steps = 0

        for _ in range(self.reps):

            reward, steps = _eval_net(net,
                                      self.env,
                                      activations=activations,
                                      seed=self.seed)

            reward_sum += reward
            total_steps += steps

        return reward_sum/self.reps, total_steps

    def eval_net_mean_novelty(self, net, activations):

        reward_sum = 0
        total_steps = 0

        # No behaviors yet
        behaviors = [None] * self.reps

        for j in range(self.reps):

            reward, behavior, steps = self.eval_net_novelty(net, activations)

            reward_sum += reward

            behaviors[j] = behavior

            total_steps += steps

        return reward_sum/self.reps, behaviors, total_steps

    def eval_net_novelty(self, net, activations):

        env = self.env
        env.seed(self.seed)
        state = env.reset()
        steps = 0

        is_discrete = _is_discrete(env)

        total_reward = 0

        while True:

            # Support recurrent nets
            for k in range(activations):
                action = net.activate(state)

            # Support both discrete and continuous actions
            action = (np.argmax(action)
                      if is_discrete
                      else action * env.action_space.high)

            state, reward, behavior, done, _ = env.step_novelty(action)

            # Accumulate reward, but not novelty
            total_reward += reward

            if done:
                break

            steps += 1

        env.close()

        # Return total reward and final behavior
        return total_reward, behavior, steps

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
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        return config.eval_net_mean(net, config.activations)


class _GymHyperConfig(_GymNeatConfig):

    def __init__(self, args, substrate=None):

        _GymNeatConfig.__init__(self, args, layout=(5, 1))

        subs = self.params['Substrate']
        actfun = subs['function']
        inp = eval(subs['input'])
        hid = eval(subs['hidden']) if substrate is None else substrate
        out = eval(subs['output'])

        self.substrate = Substrate(inp, out, hid)
        self.actfun = actfun

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
        activations = len(config.substrate.hidden_coordinates) + 2
        return config.eval_net_mean(net, activations)


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
        return config.eval_net_mean(net, esnet.activations)


class _GymPopulation(Population):
    '''
    Supports genomes that report their number of evaluations
    '''

    def __init__(self, config):

        Population.__init__(self, config)

    def run(self, fitness_function, ngen=None):

        gen = 0

        while ngen is None or gen < ngen:

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

                # Accumulate total evaluations
                self.config.current_evaluations += evaluations

                if best is None:
                    best = g

                else:
                    if g.actual_fitness > best.actual_fitness:
                        best = g

            self.reporters.post_evaluate(self.config,
                                         self.population,
                                         self.species,
                                         best)

            # Accumulate total evaluations for this run
            self.config.total_evaluations += self.config.current_evaluations

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


class _NoveltyPopulation(_GymPopulation):
    '''
    Supports genomes that report their novelty
    '''

    def __init__(self, config):

        _GymPopulation.__init__(self, config)

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

        # Ordinary report if not novelty search
        if config.novelty is None:

            StdOutReporter.post_evaluate(
                    self,
                    config,
                    population,
                    species,
                    best_genome)

        # Special report for novelty search
        else:

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

        print('Evaluations this generation: %d' % config.current_evaluations)


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
    parser.add_argument('--seed', type=int, required=False,
                        help='Seed for random number generator')
    args = parser.parse_args()

    # Default to original NEAT
    config = _GymNeatConfig(args)

    # Check for HyperNEAT, ES-HyperNEAT
    if args.hyper:
        config = _GymHyperConfig(args)
    if args.eshyper:
        config = _GymEsHyperConfig(args)

    # Set random seed (including None)
    random.seed(args.seed)

    # Make directories for saving results
    os.makedirs('models', exist_ok=True)
    os.makedirs('visuals', exist_ok=True)

    # Create an ordinary population or a population for NoveltySearch
    pop = (_NoveltyPopulation(config)
           if config.is_novelty()
           else _GymPopulation(config))

    # Add a stdout reporter to show progress in the terminal
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

    # Report total number of evaluations
    print('\nTotal evaluations = %d' % config.total_evaluations)

    # Save winner
    config.save_genome(winner)


if __name__ == '__main__':
    main()
