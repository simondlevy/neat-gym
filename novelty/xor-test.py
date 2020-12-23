#!/usr/bin/env python3
'''
Test script for testing NEAT with XOR 

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import argparse
import pickle

def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename', metavar='FILENAME', help='.dat input file')
    args = parser.parse_args()

    # Load net and environment name from pickled file
    net, task_name = pickle.load(open(args.filename, 'rb'))

    print(task_name)

    # Run the network on the environment
    #print('Reward = %6.6f.' % eval_net(net, gym.make(env_name), render=(not nodisplay), record_dir=record_dir))

if __name__ == '__main__':
    main()
