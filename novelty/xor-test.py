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

    # Load net name from pickled file
    net, _ = pickle.load(open(args.filename, 'rb'))

    # Run the network on the environment
    for inp,tgt in zip(((0,0), (0,1), (1,0), (1,1)), (0,1,1,0)):
        print(inp, net.activate(inp + (1,))[0])

if __name__ == '__main__':
    main()
