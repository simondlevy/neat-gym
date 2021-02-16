#!/usr/bin/env python3
'''
Concentric robot environment

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from neat_gym_challenges.envs.gather import GatherConcentric, gather_demo

if __name__ == '__main__':

    gather_demo(GatherConcentric())
