'''
Abstract food-gathering environment class and demo function for HyperNEAT

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import argparse
from time import sleep

import numpy as np

import gym
# from gym import spaces
from gym.utils import seeding, EzPickle


class FoodGatherConcentric(gym.Env, EzPickle):

    FRAMES_PER_SECOND = 50

    MAX_STEPS = 400

    metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': FRAMES_PER_SECOND
            }

    def __init__(self, n=8):

        EzPickle.__init__(self)
        self.seed()
        self.viewer = None
        self.n = n

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        return self.step(None)

    def step(self, action):

        return 0

    def render(self, mode='human', show_trajectory=True):

        return None

    def close(self):

        return


def demo(env):
    '''
    Runs a random-walk demo with command-line arguments.
    '''

    fmtr = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmtr)
    parser.add_argument('--seed', type=int, required=False, default=None,
                        help='Seed for random number generator')
    parser.add_argument('--steps', type=int, required=False,
                        default=FoodGatherConcentric.MAX_STEPS,
                        help='Number of steps to run')
    parser.add_argument('--traj', dest='show_trajectory',
                        action='store_true', help='Show trajectory')
    args = parser.parse_args()

    env.max_steps = args.steps
    env.seed(args.seed)
    np.random.seed(args.seed)

    state = env.reset()

    exit(0)

    for k in range(args.steps):

        action = np.random.random(2)

        state, reward, _, _ = env.step(action)

        frame = env.render(mode='rgb_array',
                           show_sensors=args.show_sensors,
                           show_trajectory=(not args.hide_trajectory))
        sleep(1./env.FRAMES_PER_SECOND)

        if frame is None:
            break

        print('step  %05d/%05d  reward = %f' %
              (k, env.max_steps, reward), end='')

    sleep(1)
    env.close()
