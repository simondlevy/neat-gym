#!/usr/bin/env python3
'''
Test script for using NEAT with OpenAI Gym environments

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from neat_gym import read_file, eval_net

if __name__ == '__main__':

    # Load genome and configuration from pickled file
    net, env, record_dir = read_file(allow_record=True)

    # Run the network
    print('%6.6f' % eval_net(net, env, render=True, record_dir=record_dir))
