#!/usr/bin/env python3
'''

Food-gathering environment class and demo function for HyperNEAT

Based on

@ARTICLE{Stanley_ahypercube-based,
    author = {Kenneth O. Stanley and David B. D'Ambrosio and Jason Gauci},
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

from pyglet.text import Label

import gym
from gym import spaces
from gym.utils import seeding, EzPickle
from gym.envs.classic_control import rendering
from gym.envs.classic_control.rendering import Transform

from neat_gym_challenges.geometry import distance_point_to_point
from neat_gym_challenges.geometry import distance_point_to_line


class _OneTimeLabel:
    '''
    https://stackoverflow.com/questions/56744840
    '''

    def __init__(self, viewer, text, **kwargs):
        self.label = Label(text, **kwargs)
        viewer.add_onetime(self)

    def render(self):
        self.label.draw()


class GatherConcentric(gym.Env, EzPickle):

    # Constants from Stanley et al.
    FOOD_DISTANCE = 100
    MAX_STEPS = 1000

    # Constants for Equation 1
    SMAX = 10
    OMAX = 10

    # Arbitrary constants
    ROBOT_RADIUS = 5
    FOOD_RADIUS = 2
    WORLD_SIZE = 400
    FRAMES_PER_SECOND = 10

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

        # Set up initial conditions for a trial
        self._restart()

        # We'll repeat for N trials
        self.trials = 0

        # Values for Equation 2
        self.r = 2 * self.n
        self.ttot = 0
        self.fc = 0

        self.need_restart = False

        # Start with a random move
        return self.step(np.random.random(self.n))[0]

    def step(self, action):

        if self.need_restart:
            self.need_restart = False
            self.trials += 1
            self.ttot += self.steps
            self._restart()

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
        self.closest = np.argmin([distance_point_to_line(self.food_location,
                                                         line)
                                  for line in self.rangefinder_lines])

        # State is a one-hot vector using the closest rangefinder
        state = np.zeros(self.n)
        state[self.closest] = 1

        # Assume no reward yet
        reward = 0
        done = False

        self.steps += 1

        # Get a reward and finish if robot is on top of food
        if (distance_point_to_point(self.robot_location,
                                    self.food_location)
           < self.ROBOT_RADIUS):
            self.fc = +1
            self.need_restart = True

        # Failed to get reward this trial
        elif self.steps == self.MAX_STEPS:
            self.need_restart = True

        # We're done when we've reached the desired number of trials
        done = (self.trials == self.r)

        # Equation 2
        reward = (10000*self.fc/self.r + 1000-self.ttot) if done else 0

        return state, reward, done, {}

    def render(self, mode='human', show_sensors=False, show_trajectory=True):

        # Avoid rendering after final trial
        if self.trials == self.r:
            return

        # Initial graphics first time around
        if self.viewer is None:

            self.viewer = rendering.Viewer(self.WORLD_SIZE, self.WORLD_SIZE)

            # Support optional display of trajectory
            self.trajectory = []

            # Set up drawing for robot
            self.robot_transform = self._make_graphic(self.ROBOT_RADIUS, False)

            # Set up drawing for food
            self.food_transform = self._make_graphic(self.FOOD_RADIUS, True)

        # Display current trial number
        _OneTimeLabel(self.viewer,
                      'Trial %03d/%03d' % (self.trials+1, self.r),
                      font_size=18,
                      x=20,
                      y=20,
                      anchor_x='left',
                      anchor_y='center',
                      color=(0, 0, 0, 255))

        # Draw food
        self.food_transform.set_translation(*self.food_location)

        # Draw robot
        self.robot_transform.set_translation(*self.robot_location)

        # Show sensors if indicated
        if show_sensors:

            for i, line in enumerate(self.rangefinder_lines):
                self._draw_line(line,
                                (1, 0, 0) if i == self.closest else (0, 1, 0))

        # Show trajectory if indicated
        if show_trajectory:

            self.trajectory.append(self.robot_location.copy())

            for i in range(len(self.trajectory)-1):
                self._draw_line((self.trajectory[i],
                                self.trajectory[i+1]),
                                (0, 0, 1))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _make_graphic(self, radius, filled):

        circle = rendering.make_circle(radius, filled=filled)
        transform = Transform(translation=(0, 0))
        circle.add_attr(transform)
        self.viewer.add_geom(circle)
        return transform

    def _restart(self):
        '''
        Sets up initial conditions for a trial
        '''

        center = np.array([self.WORLD_SIZE/2]*2)

        # Robot starts in center of room
        self.robot_location = center

        # Food starts at the angle a given sensor or between two sensors
        food_angle = (self.angles[np.random.randint(self.n)] +
                      np.random.randint(2) * self.angles[1]/2)

        # Food location is 100 units along this angle
        self.food_location = center + self._polar_to_rect(self.FOOD_DISTANCE,
                                                          food_angle)

        self.closest = None

        # Records steps in current trial
        self.steps = 0

    def _polar_to_rect(self, r, theta):

        return np.array([r * np.cos(theta), r * np.sin(theta)])

    def _draw_line(self, line, color):

        self.viewer.draw_line(line[0], line[1], color=color)

# End of Gather classes


def main():

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

    env = GatherConcentric(n=args.n)
    env.max_steps = args.steps
    env.seed(args.seed)

    state = env.reset()

    for k in range(args.steps):

        # Heuristic trick
        action = state

        state, reward, done, _ = env.step(action)

        env.render(mode='rgb_array',
                   show_sensors=(args.show_sensors),
                   show_trajectory=args.show_trajectory)

        if done:
            print('Fitness = %f' % reward)
            break

        sleep(1./env.FRAMES_PER_SECOND)

    sleep(1)
    env.close()


if __name__ == '__main__':
    main()
