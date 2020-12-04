'''
Common code for using NEAT with OpenAI Gym environments

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import neat
import gym
from gym import wrappers
import argparse
import pickle
import time
import os
from PIL import Image

class _GymConfig(neat.Config):

    def __init__(self, env_name, reps):
        '''
        env_name names environment and config file
        '''

        neat.Config.__init__(self, neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, env_name+'.cfg')

        self.env = gym.make(env_name)

        self.reps = reps

class _GymHyperConfig(_GymConfig):

    def __init__(self, env_name, reps, substrate, actfun):

        _GymConfig.__init__(self, env_name, reps)

        self.substrate = substrate
        self.actfun = actfun

def eval_net(net, env, render=False, record_dir=None):
    '''
    Evaluates an evolved network
    @param net the network
    @param env the Gym environment
    @param render set to True for rendering
    @param record_dir set to directory name for recording video
    @return total reward
    '''

    if record_dir is not None:
        env = wrappers.Monitor(env, record_dir, force=True)

    state = env.reset()
    total_reward = 0
    steps = 0

    while True:
        action = net.activate(state)
        state, reward, done, _ = env.step(action)
        if render:
            o = env.render('rgb_array')
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

    # Load genome and configuration from pickled file
    genome, config = pickle.load(open(args.filename, 'rb'))

    # Return genome, config, and optional save flag
    return genome, config, args.record if allow_record else None
