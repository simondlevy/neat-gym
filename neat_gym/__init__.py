'''
Common code for using NEAT with OpenAI Gym environments

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import neat
import gym
import argparse
import pickle

class _GymConfig(neat.Config):

    def __init__(self, env_name, reps):
        '''
        env_name names environment and config file
        '''

        neat.Config.__init__(self, neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, env_name+'.cfg')

        self.env = gym.make(env_name)

        self.reps = reps

def eval_net(net, env, render=False):
    '''
    Evaluates an evolved network
    @param net the network
    @param env the Gym environment
    @param render set to True for rendering
    @return total reward
    '''

    state = env.reset()
    rewards = 0
    steps = 0

    while True:
        action = net.activate(state)
        state, reward, done, _ = env.step(action)
        if render:
            env.render()
        rewards += reward
        steps += 1
        if done:
            break

    env.close()

    return rewards


def read_file():
    '''
    Reads a genome/config file based on command-line argument
    @return gengome,config tuple
    '''

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='FILENAME', help='input file')
    args = parser.parse_args()

    # Load genome and configuration from pickled file
    return pickle.load(open(args.filename, 'rb'))
