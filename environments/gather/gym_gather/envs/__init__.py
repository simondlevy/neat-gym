'''
Abstract food-gathering environment class and demo function for HyperNEAT

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

class FoodGather(gym.Env, EzPickle):

    FRAMES_PER_SECOND = 50

    metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': FRAMES_PER_SECOND
            }

    def __init__(self):

        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

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
                        default=Maze.MAX_STEPS, help='Number of steps to run')
    parser.add_argument('--novelty', dest='use_novelty', action='store_true',
                        help='Compute novelty')
    parser.add_argument('--sensors', dest='show_sensors', action='store_true',
                        help='Show sensors')
    parser.add_argument('--notraj', dest='hide_trajectory',
                        action='store_false', help='Hide trajectory')
    args = parser.parse_args()

    env.max_steps = args.steps
    env.seed(args.seed)
    np.random.seed(args.seed)

    state = env.reset()

    for k in range(args.steps):

        action = np.random.random(2)

        if args.use_novelty:
            state, reward, behavior, _, _ = env.step_novelty(action)
        else:
            state, reward, _, _ = env.step(action)

        frame = env.render(mode='rgb_array',
                           show_sensors=args.show_sensors,
                           show_trajectory=(not args.hide_trajectory))
        sleep(1./env.FRAMES_PER_SECOND)

        if frame is None:
            break

        if k % 20 == 0 or k == args.steps - 1:
            print('step  %05d/%05d  reward = %f' %
                  (k, env.max_steps, reward), end='')
            if args.use_novelty:
                print('  behavior =', behavior, end='')
            print()

    sleep(1)
    env.close()
