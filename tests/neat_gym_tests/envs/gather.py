'''
Food-gathering environment class and demo function for HyperNEAT

Based on

@ARTICLE{Stanley_ahypercube-based,
    author = {Kenneth O. Stanley and Jason Gauci},
    title = {A hypercube-based indirect encoding for evolving large-scale
             neural networks},
    journal = {Artificial Life},
    year = {},
    pages = {2009}
}

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import argparse
from time import sleep

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

from neat_gym_tests.geometry import distance_point_to_line


class GatherConcentric(gym.Env, EzPickle):

    ROBOT_RADIUS = 5
    FOOD_RADIUS = 2
    WORLD_SIZE = 400
    FOOD_DISTANCE = 100
    FRAMES_PER_SECOND = 50
    MAX_STEPS = 1000

    # Constants from Equation 1
    SMAX = 10
    OMAX = 10

    metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': FRAMES_PER_SECOND
            }

    def __init__(self, n=8):

        EzPickle.__init__(self)
        self.seed()
        self.viewer = None
        self.n = n

        # N sensors
        self.observation_space = spaces.Box(-np.inf,
                                            +np.inf,
                                            shape=(n,),
                                            dtype=np.float32)

        # N effecotrs
        self.action_space = spaces.Box(0, self.OMAX, (n,), dtype=np.float32)

        # Set up so that index 0 corresponds to position 1 in Figure 7a
        # self.angles = np.array(n)
        self.angles = 2*np.pi/n * np.array([n] + list(range(1, n)))

        # Set up relative endpoints for rangefinders
        self.rangefinder_points = [self._polar_to_rect(self.WORLD_SIZE, angle)
                                   for angle in self.angles]

        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        center = np.array([self.WORLD_SIZE/2]*2)

        # Robot starts in center of room
        self.robot_location = center

        # Food starts at the angle a given sensor or between two sensors
        food_angle = (self.angles[np.random.randint(self.n)] +
                      np.random.randint(2) * self.angles[1]/2)

        # Food location is 100 units along this angle
        self.food_location = center + self._polar_to_rect(self.FOOD_DISTANCE,
                                                          food_angle)

        # No rangefinder data yet
        self.rangefinder_lines = None

        # Support optional display of trajectory
        self.trajectory = []

        # Start with a random move
        return self.step(np.random.random(self.n))

    def step(self, action):

        # Pick actuator with maximum activation
        k = np.argmax(action)
        angle = self.angles[k]

        # Equation 1 (speed)
        s = (self.SMAX / self.OMAX) * (self.OMAX / np.sum(action))

        # Update location using cyclic boundary conditions
        self.robot_location = ((self.robot_location +
                                self._polar_to_rect(s, angle))
                               % self.WORLD_SIZE)

        # Update rangefinders
        self.rangefinder_lines = [(self.robot_location, self.robot_location+pt)
                                  for pt in self.rangefinder_points]

        # Get rangefinder closest to food
        print([distance_point_to_line(self.food_location, rangefinder_line)
               for rangefinder_line in self.rangefinder_lines])

        # XXX
        state = np.zeros(self.n)
        reward = 0
        done = False

        return state, reward, done, {}

    def render(self, mode='human', show_sensors=False, show_trajectory=True):

        if self.viewer is None:

            from gym.envs.classic_control import rendering
            from gym.envs.classic_control.rendering import Transform

            self.viewer = rendering.Viewer(self.WORLD_SIZE, self.WORLD_SIZE)

            # Set up drawing for robot
            self.robot_circle = rendering.make_circle(self.ROBOT_RADIUS,
                                                      filled=False)
            self.robot_transform = Transform(translation=(0, 0), rotation=0)
            self.robot_circle.add_attr(self.robot_transform)
            self.viewer.add_geom(self.robot_circle)

            # Draw food location
            self.food = rendering.make_circle(self.FOOD_RADIUS, filled=True)
            xfrm = Transform(translation=self.food_location)
            self.food.add_attr(xfrm)
            self.viewer.add_geom(self.food)

        # Show sensors if indicated
        if show_sensors:

            for line in self.rangefinder_lines:
                self._draw_line(line, (1, 0, 0))

        # Otherwise, just draw robot
        else:

            # Draw robot
            self.robot_transform.set_translation(*self.robot_location)

        # Show trajectory if indicated
        if show_trajectory:

            self.trajectory.append(self.robot_location.copy())

            for i in range(len(self.trajectory)-1):
                self._draw_line((self.trajectory[i],
                                self.trajectory[i+1]),
                                (0, 0, 1))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):

        return

    def _polar_to_rect(self, r, theta):

        return np.array([r * np.cos(theta), r * np.sin(theta)])

    def _draw_line(self, line, color):

        self.viewer.draw_line(line[0], line[1], color=color)


def gather_demo(env):
    '''
    Runs a random-walk demo with command-line arguments.
    '''

    fmtr = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmtr)
    parser.add_argument('--n', type=int, required=False, default=8,
                        help='Number of sensors (actuators)')
    parser.add_argument('--seed', type=int, required=False, default=None,
                        help='Seed for random number generator')
    parser.add_argument('--steps', type=int, required=False,
                        default=GatherConcentric.MAX_STEPS,
                        help='Number of steps to run')
    parser.add_argument('--traj', dest='show_trajectory',
                        action='store_true', help='Show trajectory')
    parser.add_argument('--sensors', dest='show_sensors',
                        action='store_true', help='Show sensors')
    args = parser.parse_args()

    env.max_steps = args.steps
    env.n = args.n
    env.seed(args.seed)

    state = env.reset()

    for k in range(args.steps):

        action = np.random.random(env.n)

        state, reward, _, _ = env.step(action)

        frame = env.render(mode='rgb_array',
                           show_sensors=args.show_sensors,
                           show_trajectory=args.show_trajectory)

        sleep(1./env.FRAMES_PER_SECOND)

        if frame is None:
            break

        last = (k == args.steps - 1)

        if k % 20 == 0 or last:
            print('step  %05d/%05d  reward = %f' % (k, env.max_steps, reward))

    sleep(1)
    env.close()
