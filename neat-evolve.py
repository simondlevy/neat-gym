#!/usr/bin/env python3
'''
NEAT evolver script for OpenAI Gym environemnts

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import argparse
from argparse import ArgumentDefaultsHelpFormatter
from neat_gym import _GymNeatConfig, _GymHyperConfig, evolve

# Parse command-line arguments
parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', default='CartPole-v1', help='Environment id')
parser.add_argument('--hyper', dest='hyper', action='store_true',
                    help='Use HyperNEAT')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true',
                    help='Save at each new best')
parser.add_argument('--cfgdir', required=False,
                    default='./config', help='Directory for config files')
parser.add_argument('--ngen', type=int, required=False,
                    help='Number of generations to run')
parser.add_argument('--reps', type=int, default=10, required=False,
                    help='Number of repetitions per genome')
parser.add_argument('--seed', type=int, required=False,
                    help='Seed for random number generator')
args = parser.parse_args()

# Default to original NEAT
configfun = _GymNeatConfig.make_config

# Check for HyperNEAT, ES-HyperNEAT
if args.hyper:
    configfun = _GymHyperConfig.make_config

config, evalfun = configfun(args)

# Evolve
evolve(config, evalfun, args.seed, args.env, args.ngen, args.checkpoint)
