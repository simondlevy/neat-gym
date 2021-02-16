#!/usr/bin/env python3
'''
Hard maze environment, based on https://github.com/yaricom/goNEAT_NS

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from neat_gym_challenges.envs import MazeHard, maze_demo


if __name__ == '__main__':

    maze_demo(MazeHard())
