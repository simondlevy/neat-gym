'''
Common code for using NEAT with OpenAI Gym environments

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import multiprocessing as mp
import os
import argparse
import random
import pickle
import time
import warnings
from configparser import ConfigParser

import gym
from gym import wrappers
import neat
import numpy as np

from neat.config import ConfigParameter, UnknownConfigItemError
from neat.population import Population, CompleteExtinctionException
from neat.genome import DefaultGenome
from neat.reporting import StdOutReporter, BaseReporter
from neat.math_util import mean, stdev

from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.es_hyperneat.es_hyperneat import ESNetwork
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate

from neat_gym.novelty import Novelty


class AugmentedGenome(DefaultGenome):
    '''
    Supports both ordinary NEAT and Novelty Search.
    '''

    def __init__(self, key):

        DefaultGenome.__init__(self, key)

        # Sparsity is used as fitness; need anoter variable for actual fitness
        self.actual_fitness = None


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
                 task_name,
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
        self.task_name = task_name
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

            try:
                names = parameters['Names']
                self.node_names = {}
                for idx, name in enumerate(eval(names['input'])):
                    self.node_names[-idx-1] = name
                for idx, name in enumerate(eval(names['output'])):
                    self.node_names[idx] = name
            except Exception:
                self.node_names = {}

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
        self.novelty = Novelty.parse(config_file_name) if novelty else None

        # Store config parameters for subclasses
        self.params = parameters

    def save_genome(self, genome):

        name = self._make_name(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, self)
        pickle.dump((net, self.task_name), open('models/%s.dat' % name, 'wb'))
        _GymNeatConfig._draw_net(net, 'visuals/%s' % name, self.node_names)

    def is_novelty(self):

        return self.novelty is not None

    def get_actual_fitness(self, genome):

        return genome.actual_fitness if self.is_novelty() else genome.fitness

    def _make_name(self, genome, suffix=''):

        return '%s%s%+010.3f' % \
               (self.task_name, suffix, self.get_actual_fitness(genome))


class _GymNeatConfig(NeatConfig):

    def __init__(self, args, layout=None):

        # Make gym environment form name in command-line arguments
        env = gym_make(args.env)

        # Make sure environment supports novelty
        if args.novelty:
            unenv = env.unwrapped
            if not hasattr(unenv, 'step_novelty'):
                print('Error: environment %s does not support novelty search' %
                      args.env)
                exit(1)

        # Get input/output layout from environment, or from layout for Hyper
        if layout is None:
            num_inputs = env.observation_space.shape[0]
            if _GymNeatConfig._is_discrete(env):
                num_outputs = env.action_space.n
            else:
                num_outputs = env.action_space.shape[0]
        else:
            num_inputs, num_outputs = layout

        # Default to environment name for config file
        cfgfilename = ('config/' + args.env + '.cfg'
                       if args.config is None else args.config)

        NeatConfig.__init__(self,
                            neat.DefaultGenome,
                            neat.DefaultReproduction,
                            neat.DefaultSpeciesSet,
                            neat.DefaultStagnation,
                            cfgfilename,
                            args.env,
                            {'num_inputs': num_inputs,
                             'num_outputs': num_outputs},
                            args.seed,
                            args.novelty)

        # Store environment, number or repetitions for later
        self.env = env
        self.reps = args.reps

    @staticmethod
    def eval_genome(genome, config):

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        return _GymNeatConfig.eval_net_mean(config, net, 1)

    @staticmethod
    def eval_net_mean(config, net, activations):

        return (_GymNeatConfig.eval_net_mean_novelty(config, net, activations)
                if config.novelty is not None
                else _GymNeatConfig.eval_net_mean_fitness(config,
                                                          net,
                                                          activations))

    @staticmethod
    def eval_net_mean_fitness(config, net, activations):

        fitness_sum = 0

        for _ in range(config.reps):

            fitness_sum += eval_net(net,
                                    config.env,
                                    activations=activations,
                                    seed=config.seed)

        return fitness_sum / config.reps

    @staticmethod
    def eval_net_mean_novelty(config, net, activations):

        fitness_sum = 0
        novelty_sum = np.zeros(config.novelty.ndims)

        for _ in range(config.reps):

            result = _GymNeatConfig._eval_net_novelty(net,
                                                      config.env,
                                                      config.novelty.ndims,
                                                      activations=activations,
                                                      seed=config.seed)
            fitness_sum += result[0]
            novelty_sum += result[1]

        return fitness_sum / config.reps, novelty_sum / config.reps

    @staticmethod
    def _eval_net_novelty(net, env, ndims, activations, seed):

        env.seed(seed)
        state = env.reset()
        total_reward = 0
        total_novelty = np.zeros(ndims)
        steps = 0

        is_discrete = _GymNeatConfig._is_discrete(env)

        while True:

            # Support recurrent nets
            for k in range(activations):
                action = net.activate(state)

            # Support both discrete and continuous actions
            action = (np.argmax(action)
                      if is_discrete else action * env.action_space.high)

            state, result, done, _ = env.step_novelty(action)

            reward, novelty = result

            total_reward += reward

            # We might only get novelty at end of episode
            #if novelty is not None:
            #    total_novelty += novelty

            if done:
                break

            steps += 1

        env.close()

        return total_reward, total_novelty

    @staticmethod
    def _draw_net(net, filename, node_names):

        # Create PDF
        draw_net(net, filename=filename, node_names=node_names)

        # Delete text
        os.remove(filename)

    def _is_discrete(env):
        return 'Discrete' in str(type(env.action_space))


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

        cppn, net = _GymHyperConfig._make_nets(genome, self)
        self._save_nets(genome, cppn, net)

    def _save_nets(self, genome, cppn, net, suffix='-hyper'):
        pickle.dump((net, self.task_name),
                    open('models/%s.dat' %
                         self._make_name(genome, suffix=suffix), 'wb'))
        _GymNeatConfig._draw_net(cppn,
                                 'visuals/%s' %
                                 self._make_name(genome, suffix='-cppn'),
                                 self.cppn_node_names)
        _GymNeatConfig._draw_net(net,
                                 'visuals/%s' %
                                 self._make_name(genome, suffix=suffix),
                                 self.node_names)

    @staticmethod
    def eval_genome(genome, config):

        cppn, net = _GymHyperConfig._make_nets(genome, config)
        activations = len(config.substrate.hidden_coordinates) + 2
        return _GymNeatConfig.eval_net_mean(config, net, activations)

    @staticmethod
    def _make_nets(genome, config):

        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        return (cppn,
                create_phenotype_network(cppn,
                                         config.substrate,
                                         config.actfun))


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

        cppn, _, net = _GymEsHyperConfig._make_nets(genome, self)
        _GymEsHyperConfig._save_nets(self, genome, cppn, net,
                                     suffix='-eshyper')

    @staticmethod
    def eval_genome(genome, config):

        _, esnet, net = _GymEsHyperConfig._make_nets(genome, config)
        return _GymNeatConfig.eval_net_mean(config, net, esnet.activations)

    @staticmethod
    def _make_nets(genome, config):

        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        esnet = ESNetwork(config.substrate, cppn, config.es_params)
        net = esnet.create_phenotype_network()
        return cppn, esnet, net


class _NoveltyPopulation(Population):

    def __init__(self, config):

        Population.__init__(self, config)

    def run(self, fitness_function, n=None):

        k = 0

        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation)

            # Evaluate all genomes using the user-provided function.
            fitness_function(list(self.population.items()), self.config)

            # Gather and report statistics.
            best = None
            for g in self.population.values():
                if g.fitness is None:
                    raise RuntimeError('Fitness not assigned to genome %d' %
                                       g.key)

                # Use actual_fitness to encode ignored objective,
                # and replace genome's fitness with its novelty
                g.actual_fitness, novelty = g.fitness
                g.fitness = self.config.novelty.add(novelty)

                if best is None:
                    best = g

                else:
                    print(g.actual_fitness, best.actual_fitness)
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
            self._reproduce()

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

    def _reproduce(self):
        self.population = \
                 self.reproduction.reproduce(self.config, self.species,
                                             self.config.pop_size,
                                             self.generation)

    def _create_new_pop(self):
        self.population = \
                self.reproduction.create_new(self.config.genome_type,
                                             self.config.genome_config,
                                             self.config.pop_size)


class _SaveReporter(BaseReporter):

    def __init__(self, task_name, checkpoint):

        BaseReporter.__init__(self)

        self.best_fitness = -np.inf
        self.task_name = task_name
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
        fitnesses = [c.fitness for c in population.values()]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        print('Population\'s average novelty: %3.5f stdev: %3.5f' %
              (fit_mean, fit_std))
        print('Best novelty: %3.5f - size: (%d,%d) - species %d - id %d' %
              (best_genome.fitness,
               best_genome.size()[0],
               best_genome.size()[1],
               best_species_id,
               best_genome.key))
        print('Best actual fitness: %f ' % best_genome.actual_fitness)

# Public functions ===================================================


def gym_make(envname):

    env = None

    try:
        env = gym.make(envname)

    except Exception:
        print('Unable to make environment %s [check name or __init__()]' %
              envname)
        exit(1)

    return env


def evolve(config, evalfun, seed, task_name, ngen, checkpoint):
    '''
    NEAT evolution with parallel evaluator
    '''

    # Set random seed (including None)
    random.seed(seed)

    # Make directories for saving results
    os.makedirs('models', exist_ok=True)
    os.makedirs('visuals', exist_ok=True)

    # Create an ordinary population or a population for NoveltySearch
    pop = (_NoveltyPopulation(config)
           if config.is_novelty() else Population(config))

    # Add a stdout reporter to show progress in the terminal.
    pop.add_reporter(_StdOutReporter(show_species_detail=False))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Add a reporter (which can also checkpoint the best)
    pop.add_reporter(_SaveReporter(task_name, checkpoint))

    # Create a parallel fitness evaluator
    pe = neat.ParallelEvaluator(mp.cpu_count(), evalfun)

    # Run for number of generations specified in config file
    winner = (pop.run(pe.evaluate)
              if ngen is None else pop.run(pe.evaluate, ngen))

    # Save winner
    config.save_genome(winner)


def read_file(allow_record=False):
    '''
    Reads a genome/config file based on command-line argument
    @return genome,config tuple
    '''

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename', metavar='FILENAME', help='.dat input file')
    parser.add_argument('--nodisplay', dest='nodisplay', action='store_true',
                        help='Suppress display')
    if allow_record:
        parser.add_argument('--record', default=None,
                            help='If specified, sets the recording dir')
    args = parser.parse_args()

    # Load net and environment name from pickled file
    net, env_name = pickle.load(open(args.filename, 'rb'))

    # Return genome, config, and optional save flag
    return net, env_name, args.record if allow_record else None, args.nodisplay


def eval_net(
        net,
        env,
        render=False,
        record_dir=None,
        activations=1,
        seed=None):
    '''
    Evaluates a network
    @param net the network
    @param env the Gym environment
    @param render set to True for rendering
    @param record_dir set to directory name for recording video
    @param activations number of times to repeat
    @param seed seed for random number generator
    @return total reward
    '''

    if record_dir is not None:
        env = wrappers.Monitor(env, record_dir, force=True)

    env.seed(seed)
    state = env.reset()
    total_reward = 0
    steps = 0

    is_discrete = _GymNeatConfig._is_discrete(env)

    while True:

        # Support recurrent nets
        for k in range(activations):
            action = net.activate(state)

        # Support both discrete and continuous actions
        action = (np.argmax(action)
                  if is_discrete else action * env.action_space.high)

        state, reward, done, _ = env.step(action)

        if render:
            env.render('rgb_array')
            time.sleep(.02)

        total_reward += reward

        if done:
            break

        steps += 1

    env.close()

    return total_reward
