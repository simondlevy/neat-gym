#!/usr/bin/env python3
'''
Test script for using NEAT with OpenAI Gym environments

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import neat
from neat_gym import read_file, eval_net

if __name__ == '__main__':

    # Load genome and configuration from pickled file
    genome, config, savedir = read_file(save=True)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Run the network
    print('%6.6f' % eval_net(net, config.env, render=True, savedir=savedir))
