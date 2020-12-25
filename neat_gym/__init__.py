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

class NoveltyGenome(DefaultGenome):

    def __init__(self, key):

        DefaultGenome.__init__(self, key)

        # Since sparsity is used as fitness, we need a separate variable to store actual fitness
        self.actual_fitness = None

class NeatConfig(object):
    #Adapted from https://github.com/CodeReclaimers/neat-python/blob/master/neat/config.py

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
            seed):

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
            raise Exception('No such config file: ' + os.path.abspath(config_file_name))

        parameters = ConfigParser()
        with open(config_file_name) as f:
            if hasattr(parameters, 'read_file'):
                parameters.read_file(f)
            else:
                parameters.readfp(f)

            try:
                names =  parameters['Names']
                self.node_names = {}
                for idx,name in enumerate(eval(names['input'])):
                    self.node_names[-idx-1] = name
                for idx,name in enumerate(eval(names['output'])):
                    self.node_names[idx] = name
            except:
                self.node_names = {}

        # NEAT configuration
        if not parameters.has_section('NEAT'):
            raise RuntimeError("'NEAT' section not found in NEAT configuration file.")

        param_list_names = []
        for p in self.__params:
            if p.default is None:
                setattr(self, p.name, p.parse('NEAT', parameters))
            else:
                try:
                    setattr(self, p.name, p.parse('NEAT', parameters))
                except Exception:
                    setattr(self, p.name, p.default)
                    warnings.warn("Using default {!r} for '{!s}'".format(p.default, p.name),
                                  DeprecationWarning)
            param_list_names.append(p.name)
        param_dict = dict(parameters.items('NEAT'))
        unknown_list = [x for x in param_dict if x not in param_list_names]
        if unknown_list:
            if len(unknown_list) > 1:
                raise UnknownConfigItemError("Unknown (section 'NEAT') configuration items:\n" +
                                             "\n\t".join(unknown_list))
            raise UnknownConfigItemError(
                "Unknown (section 'NEAT') configuration item {!s}".format(unknown_list[0]))

        # Parse type sections.
        genome_dict = dict(parameters.items(genome_type.__name__))

        # Add layout (input/output) info
        for key in layout_dict:
            genome_dict[key] = layout_dict[key]

        self.genome_config = genome_type.parse_config(genome_dict)

        species_set_dict = dict(parameters.items(species_set_type.__name__))
        self.species_set_config = species_set_type.parse_config(species_set_dict)

        stagnation_dict = dict(parameters.items(stagnation_type.__name__))
        self.stagnation_config = stagnation_type.parse_config(stagnation_dict)

        reproduction_dict = dict(parameters.items(reproduction_type.__name__))
        self.reproduction_config = reproduction_type.parse_config(reproduction_dict)

        # Support novelty search
        self.novelty = None
        if parameters.has_section('Novelty'):
            novelty = parameters['Novelty']
            self.novelty = Novelty(
                    int(novelty['k']), 
                    float(novelty['threshold']), 
                    int(novelty['limit']), 
                    int(novelty['ndims']))

    def save_genome(self, genome):

        name = self._make_name(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, self)
        pickle.dump((net, self.task_name), open('models/%s.dat' % name, 'wb'))
        _GymNeatConfig._draw_net(net, 'visuals/%s'%name, self.node_names)

    def _make_name(self, genome, suffix=''):

        return '%s%s%+010.3f' % (self.task_name, suffix, genome.fitness)

class _NoveltyPopulation(Population):
    #Adapted from https://github.com/CodeReclaimers/neat-python/blob/master/neat/population.py

    def __init__(self, config):

        neat.Population.__init__(self, config)

    def run(self, fitness_function, n=None):

        k = 0
        #best_actual_fitness = -np.inf

        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation)

            # Evaluate all genomes using the user-provided function.
            fitness_function(list(self.population.items()), self.config)

            # Gather and report statistics.
            best = None
            for g in self.population.values():
                if g.fitness is None:
                    raise RuntimeError("Fitness not assigned to genome {}".format(g.key))

                # Use actual_fitness to encode actual fitness, and replace genome's fitnss with its novelty
                behavior, g.actual_fitness = g.fitness
                g.fitness = self.config.novelty.add(behavior)                

                if best is None or g.actual_fitness > best.actual_fitness:
                    best = g
            
                #if g.actual_fitness > best_actual_fitness:
                #    best_actual_fitness = g.actual_fitness
                #    print('******************************************** ', best_actual_fitness)

            self.reporters.post_evaluate(self.config, self.population, self.species, best)

            # Track the best genome ever seen.
            if self.best_genome is None or best.actual_fitness > self.best_genome.actual_fitness:
                self.best_genome = best

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.actual_fitness for g in self.population.values())
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)

            self.reporters.end_generation(self.config, self.population, self.species)

            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        return self.best_genome


class _GymNeatConfig(NeatConfig):

    def __init__(self, args, layout_dict, suffix=''):

        filename = args.cfgdir + '/' + args.env + suffix + '.cfg'

        NeatConfig.__init__(self, neat.DefaultGenome, neat.DefaultReproduction,
                neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                filename, args.env, layout_dict, args.seed)

        self.env = gym.make(args.env)

        self.reps = args.reps

    @staticmethod
    def eval_genome(genome, config):

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        return _GymNeatConfig.eval_net_mean(config, net, 1)

    @staticmethod
    def eval_net_mean(config, net, activations):

        fitness = 0

        for _ in range(config.reps):

            fitness += eval_net(net, config.env, activations=activations, seed=config.seed)

        return fitness / config.reps

    @staticmethod
    def load(args, suffix):

        filename = args.cfgdir + '/' + args.env + suffix + '.cfg'
        if not os.path.isfile(filename):
            print('Cannot open config file ' + filename)
            exit(1)

        parser = ConfigParser()
        parser.read(filename)
        return parser

    @staticmethod 
    def _draw_net(net, filename, node_names):

        # Create PDF
        draw_net(net, filename=filename, node_names=node_names) 

        # Delete text
        os.remove(filename) 

    @staticmethod
    def make_config(args):

        # Get input/output layout from environment
        env = gym.make(args.env)
        num_inputs  = env.observation_space.shape[0]
        num_outputs = env.action_space.n if _GymNeatConfig._is_discrete(env) else env.action_space.shape[0]

        # Load rest of config from file
        config = _GymNeatConfig(args, {'num_inputs':num_inputs, 'num_outputs':num_outputs})
        evalfun = _GymNeatConfig.eval_genome
     
        return config, evalfun

    def _is_discrete(env):
        return 'Discrete' in str(type(env.action_space))


class _GymHyperConfig(_GymNeatConfig):

    def __init__(self, args, substrate, actfun, suffix='-hyper'):

        _GymNeatConfig.__init__(self, args, {'num_inputs':5, 'num_outputs':1}, suffix)

        self.substrate = substrate
        self.actfun = actfun

        # Output of CPPN is recurrent, so negate indices
        self.node_names = {j:self.node_names[k] for j,k in enumerate(self.node_names)} 

        # CPPN itself always has the same input and output nodes XXX are these correct?
        self.cppn_node_names = {-1:'x1', -2:'y1', -3:'x2', -4:'y2', -5:'bias', 0:'weight'}

    def save_genome(self, genome):

        cppn, net = _GymHyperConfig._make_nets(genome, self)
        self._save_nets(genome, cppn, net)

    def _save_nets(self, genome, cppn, net, suffix='-hyper'):
        pickle.dump((net, self.task_name), open('models/%s.dat' % self._make_name(genome, suffix=suffix), 'wb'))
        _GymNeatConfig._draw_net(cppn, 'visuals/%s' % self._make_name(genome, suffix='-cppn'), self.cppn_node_names)
        _GymNeatConfig._draw_net(net, 'visuals/%s' % self._make_name(genome, suffix=suffix), self.node_names)

    @staticmethod
    def eval_genome(genome, config): # _GymHyperConfig

        cppn, net = _GymHyperConfig._make_nets(genome, config)
        activations = len(config.substrate.hidden_coordinates) + 2
        return _GymNeatConfig.eval_net_mean(config, net, activations)

    @staticmethod
    def _make_nets(genome, config):

        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        return cppn, create_phenotype_network(cppn, config.substrate, config.actfun)

    @staticmethod
    def make_config(args):
        
        cfg = _GymNeatConfig.load(args, '-hyper')
        subs =  cfg['Substrate']
        actfun = subs['function']
        inp = eval(subs['input'])
        hid = eval(subs['hidden'])
        out = eval(subs['output'])
        substrate = Substrate(inp, out, hid)

        # Load rest of config from file
        config = _GymHyperConfig(args, substrate, actfun)

        evalfun = _GymHyperConfig.eval_genome

        return config, evalfun
     
class _GymEsHyperConfig(_GymHyperConfig):

    def __init__(self, args, substrate, actfun, params):

        self.params = {
                'initial_depth'     : int(params['initial_depth']),
                'max_depth'         : int(params['max_depth']),
                'variance_threshold': float(params['variance_threshold']),  
                'band_threshold'    : float(params['band_threshold']),  
                'iteration_level'   : int(params['iteration_level']),  
                'division_threshold': float(params['division_threshold']),  
                'max_weight'        : float(params['max_weight']),
                'activation'        : params['activation']  
                }

        _GymHyperConfig.__init__(self, args, substrate, actfun, suffix='-eshyper')

    def save_genome(self, genome):

        cppn, _, net = _GymEsHyperConfig._make_nets(genome, self)
        _GymEsHyperConfig._save_nets(self, genome, cppn, net, suffix='-eshyper')

    @staticmethod
    def eval_genome(genome, config):

        _, esnet, net = _GymEsHyperConfig._make_nets(genome, config)
        return _GymNeatConfig.eval_net_mean(config, net, esnet.activations)

    @staticmethod
    def _make_nets(genome, config):

        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        esnet = ESNetwork(config.substrate, cppn, config.params)
        net = esnet.create_phenotype_network()
        return cppn, esnet, net

    @staticmethod
    def make_config(args):

        # Load config from file
        cfg = _GymNeatConfig.load(args, '-eshyper')
        subs =  cfg['Substrate']
        actfun = subs['function']
        inp = eval(subs['input'])
        out = eval(subs['output'])

        # Get substrate from -hyper.cfg file named by Gym environment
        substrate = Substrate(inp, out)

        # Load rest of config from file
        config = _GymEsHyperConfig(args, substrate, actfun, cfg['ES'])

        evalfun = _GymEsHyperConfig.eval_genome

        return config, evalfun

class _SaveReporter(BaseReporter):

    def __init__(self, task_name, checkpoint):

        BaseReporter.__init__(self)

        self.best = None
        self.task_name = task_name
        self.checkpoint = checkpoint

    def post_evaluate(self, config, population, species, best_genome):

        if self.checkpoint and (self.best is None or best_genome.fitness > self.best):
            self.best = best_genome.fitness
            print('############# Saving new best %f ##############' % self.best)
            config.save_genome(best_genome)

class _StdOutReporter(StdOutReporter):

    def __init__(self, show_species_detail):

        StdOutReporter.__init__(self, show_species_detail)

    def post_evaluate(self, config, population, species, best_genome):
        if config.novelty is None:
            StdOutReporter.post_evaluate(self, config, population, species, best_genome)
            return
        print('Best actual fitness: %f ' % best_genome.actual_fitness)
        fitnesses = [c.fitness for c in population.values()]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        print('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
        print('Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
            best_genome.size(),
            best_species_id,
            best_genome.key))

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
    pop = _NoveltyPopulation(config) if config.novelty is not None else neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    pop.add_reporter(_StdOutReporter(show_species_detail=False))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
    # Add a reporter (which can also checkpoint the best)
    pop.add_reporter(_SaveReporter(task_name, checkpoint))

    # Create a parallel fitness evaluator
    pe = neat.ParallelEvaluator(mp.cpu_count(), evalfun)

    # Run for number of generations specified in config file
    winner = pop.run(pe.evaluate) if ngen is None else pop.run(pe.evaluate, ngen) 

    # Save winner
    config.save_genome(winner)

def _evolve_gym(configfun):
    '''
    Evolves solutions to Gym environments based on command-line arguments
    '''

    # Parse command-line arguments

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='CartPole-v1', help='Environment id')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Save at each new best')
    parser.add_argument('--cfgdir', required=False, default='./config', help='Directory for config files')
    parser.add_argument('--ngen', type=int, required=False, help='Number of generations to run')
    parser.add_argument('--reps', type=int, default=10, required=False, help='Number of repetitions per genome')
    parser.add_argument('--seed', type=int, required=False, help='Seed for random number generator')
    args = parser.parse_args()

    # Get configuration and genome evaluation function for a particular algorithm
    config, evalfun = configfun(args) 

    # Evolve
    evolve(config, evalfun, args.seed, args.env, args.ngen, args.checkpoint)

def read_file(allow_record=False):
    '''
    Reads a genome/config file based on command-line argument
    @return genome,config tuple
    '''

    # Parse command-line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename', metavar='FILENAME', help='.dat input file')
    parser.add_argument('--nodisplay', dest='nodisplay', action='store_true', help='Suppress display')
    if allow_record:
        parser.add_argument('--record', default=None, help='If specified, sets the recording dir')
    args = parser.parse_args()

    # Load net and environment name from pickled file
    net, env_name = pickle.load(open(args.filename, 'rb'))

    # Return genome, config, and optional save flag
    return net, env_name, args.record if allow_record else None, args.nodisplay

def eval_net(net, env, render=False, record_dir=None, activations=1, seed=None):
    '''
    Evaluates an evolved network
    @param net the network
    @param env the Gym environment
    @param render set to True for rendering
    @param record_dir set to directory name for recording video
    @param actviations number of times to repeat
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
        action = np.argmax(action) if is_discrete else action * env.action_space.high

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
