#!/usr/bin/env python3
'''
Test script for using NEAT with OpenAI Gym environments

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from common import read_file, eval_genome

if __name__ == '__main__':

     # Load genome and configuration from pickled file
    genome, config = read_file()

    # Training uses multiple repetitions, testing only one
    config.reps = 1 
    print('%6.6f' % eval_genome(genome, config, True))
