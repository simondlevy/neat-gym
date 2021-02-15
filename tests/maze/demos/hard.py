#!/usr/bin/env python3
'''
Hard maze environment, based on https://github.com/yaricom/goNEAT_NS

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from gym_nsmaze.envs import Hard, demo


if __name__ == '__main__':

    demo(Hard())
