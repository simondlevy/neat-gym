'''
Common code for using NEAT with OpenAI Gym environments

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import argparse
import pickle
import time
import numpy as np
from configparser import ConfigParser
import neat
import gym
from gym import wrappers
from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.shared.visualize import draw_net

class _GymConfig(neat.Config):

    def __init__(self, args, suffix='cfg'):
        '''
        env_name names environment and config file
        '''

        neat.Config.__init__(self, neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                         args.cfgdir + '/' + args.env+'.' + suffix)

        self.env_name = args.env
        self.env = gym.make(args.env)

        self.reps = args.reps
        self.seed = args.seed

        # Open nodenames file if available
        try:
            namescfg = _read_config(args, 'names')
            names =  namescfg['Names']
            self.node_names = {}
            for idx,name in enumerate(eval(names['input'])):
                self.node_names[-idx-1] = name
            for idx,name in enumerate(eval(names['output'])):
                self.node_names[idx] = name
        except:
            self.node_names = {}

        # Output of CPPN is recurrent, so negate indices
        if args.hyper:
            self.node_names = {j:self.node_names[k] for j,k in enumerate(self.node_names)} 

    def save_genome(self, genome):

        name = self._make_name(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, self)
        pickle.dump((net, self.env), open('models/%s.dat' % name, 'wb'))
        draw_net(net, filename='visuals/%s'%name)

    def _make_name(self, genome, suffix=''):

        return '%s%s%+f' % (self.env_name, suffix, genome.fitness)

class _GymHyperConfig(_GymConfig):

    def __init__(self, args, substrate, actfun):

        _GymConfig.__init__(self, args, 'cppn')

        self.substrate = substrate
        self.actfun = actfun

    def save_genome(self, genome):

        cppn = neat.nn.FeedForwardNetwork.create(genome, self)
        net = create_phenotype_network(cppn, self.substrate)
        pickle.dump((net, self.env), open('models/%s.dat' % self._make_name(genome, suffix='-hyper'), 'wb'))
        draw_net(cppn, filename='visuals/%s' % self._make_name(genome, suffix='-cppn'))
        draw_net(net, filename='visuals/%s' % self._make_name(genome, suffix='-hyper'), node_names=self.node_names)

def _read_config(args, ext):

    parser = ConfigParser()
    parser.read(args.cfgdir + '/' + args.env + '.' + ext)
    return parser

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

    # Load net and environment from pickled file
    net, env = pickle.load(open(args.filename, 'rb'))

    # Return genome, config, and optional save flag
    return net, env, args.record if allow_record else None
