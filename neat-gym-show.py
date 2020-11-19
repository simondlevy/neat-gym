#!/usr/bin/env python3
'''
Network-display script for Lander-v2

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from neat_gym import read_file, visualize

if __name__ == '__main__':

    genome, config = read_file()

    node_names = {-1:'x', -2:'dx', -3:'y', -4:'dy', -5:'phi', -6:'dphi', 0:'mr', 1:'ml'}
    visualize.draw_net(config, genome, True, node_names = node_names)
