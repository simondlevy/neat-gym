'''
Common code for using NEAT with OpenAI Gym environments Test script for using NEAT with OpenAI Gym environments

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import neat
import gym
import numpy as np
import argparse
import pickle

class GymConfig(neat.Config):

    def __init__(self, env_name, reps):
        '''
        env_name names environment and config file
        '''

        neat.Config.__init__(self, neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, env_name+'.cfg')

        self.env = gym.make(env_name)

        self.reps = reps

def eval_genome(genome, config, render=False):

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitness = 0

    for _ in range(config.reps):

        state = config.env.reset()
        rewards = 0
        steps = 0

        while True:
            action = np.clip(net.activate(state), -1, +1)
            state, reward, done, _ = config.env.step(action)
            if render:
                config.env.render()
            rewards += reward
            steps += 1
            if done:
                break

        fitness += rewards

    config.env.close()

    return fitness / config.reps

def read_file():

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='FILENAME', help='input file')
    args = parser.parse_args()

    # Load genome and configuration from pickled file
    return pickle.load(open(args.filename, 'rb'))


