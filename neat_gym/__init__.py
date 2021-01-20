'''
Common code for using NEAT with OpenAI Gym environments

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import argparse
import pickle
import time

import gym
from gym import wrappers
import numpy as np


def _gym_make(envname):

    env = None

    try:
        env = gym.make(envname)

    except Exception as e:
        print('Unable to make environment %s: %s' %
              (envname, e))
        exit(1)

    return env


def _is_discrete(env):
    return 'Discrete' in str(type(env.action_space))


# Public functions ===================================================


def read_file(allow_record=False):
    '''
    Reads a genome/config file based on command-line argument
    @param allow_record set to True to support recording to movie
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
        report=False,
        record_dir=None,
        activations=1,
        seed=None,
        max_episode_steps=None):
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

    is_discrete = _is_discrete(env)

    while max_episode_steps is None or steps < max_episode_steps:

        # Support recurrent nets
        for k in range(activations):
            action = net.activate(state)

        # Support both discrete and continuous actions
        action = (np.argmax(action)
                  if is_discrete else action * env.action_space.high)

        state, reward, done, _ = env.step(action)

        #if render or (record_dir is not None):
        if render or (record_dir is not None):
            #env.render('rgb_array')
            env.render()
            time.sleep(.02)

        total_reward += reward

        if done:
            break

        steps += 1

    env.close()

    if report:
        print('Got reward %+6.6f in %d steps' % (total_reward, steps))

    return total_reward, steps
