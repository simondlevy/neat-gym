'''
Common code for using NEAT with OpenAI Gym environments

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import argparse
import pickle
import time
import os
import warnings
from configparser import ConfigParser

import numpy as np

import neat
from neat.config import ConfigParameter, UnknownConfigItemError

import gym
from gym import wrappers

from pureples.hyperneat.hyperneat import create_phenotype_network
#from pureples.es_hyperneat.es_hyperneat import ESNetwork
from pureples.shared.visualize import draw_net

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

def _eval_genome_eshyper(genome, config):

    #cppn = neat.nn.FeedForwardNetwork.create(genome, config)
    #esnet = ESNetwork(config.substrate, cppn, params)
    #net = esnet.create_phenotype_network()

    #activations = len(config.substrate.hidden_coordinates) + 2

    #return _eval_genome(genome, config, net, activations)
    return 0

class _Config(object):
    #Adapted from https://github.com/CodeReclaimers/neat-python/blob/master/neat/config.py

    __params = [ConfigParameter('pop_size', int),
                ConfigParameter('fitness_criterion', str),
                ConfigParameter('fitness_threshold', float),
                ConfigParameter('reset_on_extinction', bool),
                ConfigParameter('no_fitness_termination', bool, False)]

    def __init__(self, genome_type, reproduction_type, species_set_type, stagnation_type, filename, cppn_dict):

        # Check that the provided types have the required methods.
        assert hasattr(genome_type, 'parse_config')
        assert hasattr(reproduction_type, 'parse_config')
        assert hasattr(species_set_type, 'parse_config')
        assert hasattr(stagnation_type, 'parse_config')

        self.genome_type = genome_type
        self.reproduction_type = reproduction_type
        self.species_set_type = species_set_type
        self.stagnation_type = stagnation_type

        if not os.path.isfile(filename):
            raise Exception('No such config file: ' + os.path.abspath(filename))

        parameters = ConfigParser()
        with open(filename) as f:
            if hasattr(parameters, 'read_file'):
                parameters.read_file(f)
            else:
                parameters.readfp(f)

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

        # Add CPPN inputs/output (always 5/1) if specified 
        for key in cppn_dict:
            genome_dict[key] = cppn_dict[key]

        self.genome_config = genome_type.parse_config(genome_dict)

        species_set_dict = dict(parameters.items(species_set_type.__name__))
        self.species_set_config = species_set_type.parse_config(species_set_dict)

        stagnation_dict = dict(parameters.items(stagnation_type.__name__))
        self.stagnation_config = stagnation_type.parse_config(stagnation_dict)

        reproduction_dict = dict(parameters.items(reproduction_type.__name__))
        self.reproduction_config = reproduction_type.parse_config(reproduction_dict)

class _GymConfig(_Config):

    def __init__(self, args, suffix='', cppn_dict={}):
        '''
        env_name names environment and config file
        '''

        filename = args.cfgdir + '/' + args.env + suffix + '.cfg'

        if not os.path.isfile(filename):
            print('Unable to open config file ' + filename)
            exit(1)

        _Config.__init__(self, neat.DefaultGenome, neat.DefaultReproduction,
                neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                filename, cppn_dict)

        self.env_name = args.env
        self.env = gym.make(args.env)

        self.reps = args.reps
        self.seed = args.seed

        namescfg = _GymConfig.load(args, suffix)

        try:
            names =  namescfg['Names']
            self.node_names = {}
            for idx,name in enumerate(eval(names['input'])):
                self.node_names[-idx-1] = name
            for idx,name in enumerate(eval(names['output'])):
                self.node_names[idx] = name
        except:
            self.node_names = {}

    @staticmethod
    def load(args, suffix):

        filename = args.cfgdir + '/' + args.env + suffix + '.cfg'
        if not os.path.isfile(filename):
            print('Cannot open config file ' + filename)
            exit(1)

        parser = ConfigParser()
        parser.read(filename)
        return parser

    def save_genome(self, genome):

        name = self._make_name(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, self)
        pickle.dump((net, self.env_name), open('models/%s.dat' % name, 'wb'))
        draw_net(net, filename='visuals/%s'%name, node_names=self.node_names)

    def _make_name(self, genome, suffix=''):

        return '%s%s%+010.3f' % (self.env_name, suffix, genome.fitness)

class _GymHyperConfig(_GymConfig):

    def __init__(self, args, substrate, actfun, suffix='-hyper'):

        _GymConfig.__init__(self, args, suffix, {'num_inputs':5, 'num_outputs':1})

        self.substrate = substrate
        self.actfun = actfun

        # Output of CPPN is recurrent, so negate indices
        self.node_names = {j:self.node_names[k] for j,k in enumerate(self.node_names)} 

        # CPPN itself always has the same input and output nodes XXX are these correct?
        self.cppn_node_names = {-1:'x1', -2:'y1', -3:'x2', -4:'y2', -5:'bias', 0:'weight'}

    def save_genome(self, genome):

        cppn = neat.nn.FeedForwardNetwork.create(genome, self)
        net = create_phenotype_network(cppn, self.substrate)
        pickle.dump((net, self.env_name), open('models/%s.dat' % self._make_name(genome, suffix='-hyper'), 'wb'))
        draw_net(cppn, filename='visuals/%s' % self._make_name(genome, suffix='-cppn'), node_names=self.cppn_node_names)
        draw_net(net, filename='visuals/%s' % self._make_name(genome, suffix='-hyper'), node_names=self.node_names)

class _GymEsHyperConfig(_GymHyperConfig):

    def __init__(self, args, substrate, actfun):

        _GymHyperConfig.__init__(self, args, substrate, actfun, suffix='-eshyper')

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

    is_discrete = 'Discrete' in str(type(env.action_space))

    while True:
        for k in range(activations): # Support recurrent nets
            action = net.activate(state)
        if is_discrete:
            action = np.argmax(action)
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


def read_file(allow_record=False):
    '''
    Reads a genome/config file based on command-line argument
    @return genome,config tuple
    '''

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='FILENAME', help='input file')
    if allow_record:
        parser.add_argument('--record', default=None, help='If specified, sets the recording dir')
    args = parser.parse_args()

    # Load net and environment name from pickled file
    net, env_name = pickle.load(open(args.filename, 'rb'))

    # Return genome, config, and optional save flag
    return net, env_name, args.record if allow_record else None
