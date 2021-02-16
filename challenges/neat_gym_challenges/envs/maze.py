'''
Abstract Maze environment class and demo function for Novelty Search

Adapted from
https:#github.com/yaricom/goNEAT_NS/blob/master/experiments/maze/environment.go

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from time import sleep
import argparse

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

from neat_gym_challenges.geometry import distance_point_to_point
from neat_gym_challenges.geometry import distance_point_to_line
from neat_gym_challenges.geometry import rotate, intersection, arctan_degrees


class Maze(gym.Env, EzPickle):

    ROBOT_RADIUS = 5
    EXIT_RADIUS = 2

    RANGEFINDER_RANGE = 100
    RANGEFINDER_ANGLES = -90.0, -45.0, 0.0, 45.0, 90.0, -180.0
    RADAR_ANGLES = 315, 45, 135, 225

    MAX_SPEED = 3
    FRAMES_PER_SECOND = 50

    MAX_STEPS = 400

    metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': FRAMES_PER_SECOND
            }

    def __init__(self,
                 initial_location,
                 initial_heading,
                 exit_location,
                 walls):

        EzPickle.__init__(self)
        self.seed()
        self.viewer = None
        self.exit_location = exit_location
        self.walls = walls
        self.novelty = False

        # Supports changing max steps for demo
        self.max_steps = Maze.MAX_STEPS

        # Infer maze size from walls
        walls = np.reshape(walls, (2*len(self.WALLS), 2))
        marg = min(walls.flatten())
        self.maze_size = 2*(max(walls[:, 0])+marg), 2*(max(walls[:, 1])+marg)

        # Ten observations plus bias
        self.observation_space = spaces.Box(-np.inf,
                                            +np.inf,
                                            shape=(10,),
                                            dtype=np.float32)

        # Left/right, forward/back
        self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)

        self.location = np.array(initial_location)
        self.heading = initial_heading
        self.linear_velocity = 0
        self.angular_velocity = 0

        self.rangefinders = np.zeros(len(self.RANGEFINDER_ANGLES))
        self.radar = np.zeros(len(self.RADAR_ANGLES))

        # For rendering
        self.rangefinder_lines = [None]*len(self.RANGEFINDER_ANGLES)

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        self.steps = 0

        self.prev_shaping = None

        # 0.5 is halfway along logistic sigmoid
        return self.step(np.array([0.5, 0.5]))[0]

    def step(self, action):

        # Convert action to angular, linear velocities
        self.angular_velocity = self._update_velocity(self.angular_velocity,
                                                      action[0])
        self.linear_velocity = self._update_velocity(self.linear_velocity,
                                                     action[1])

        # Integrate angular velocity to get heading, keeping it in [0,360]
        self.heading = (self.heading + self.angular_velocity) % 360

        # Find next location
        vx = np.cos(np.radians(self.heading)) * self.linear_velocity
        vy = np.sin(np.radians(self.heading)) * self.linear_velocity
        newloc = vx + self.location[0], vy + self.location[1]

        # Move to new location if it wouldn't cause a collision
        if not self._collision(newloc):
            self.location = np.array(newloc)

        # Update sensors
        self._update_rangefinders()
        self._update_radar()

        # Compute distance from exit
        distance_to_exit = distance_point_to_point(self.location,
                                                   self.exit_location)

        # We're done if we reach the maximum allowed steps
        done = self.steps == self.max_steps
        self.steps += 1

        # Only get reward and location on on last step
        reward = -distance_to_exit if done else 0
        location = tuple(self.location) if done else None

        # State (observation) is rangefinders and radar
        state = np.append(self.rangefinders, self.radar)

        # Support Novelty Search
        info = {'behavior': location}

        return state, reward, done, info

    def render(self, mode='human', show_sensors=False, show_trajectory=True):

        # Helper class for showing radar as cardinal direction
        class _DrawText:

            def __init__(self, label):
                self.label = label

            def render(self):
                self.label.draw()

        if self.viewer is None:

            from gym.envs.classic_control import rendering
            from gym.envs.classic_control.rendering import Transform

            self.viewer = rendering.Viewer(*self.maze_size)

            # Set up drawing for robot
            self.robot_circle = rendering.make_circle(self.ROBOT_RADIUS,
                                                      filled=False)
            self.robot_line = self.viewer.draw_line((0, 0),
                                                    (self.ROBOT_RADIUS, 0))
            self.robot_transform = Transform(translation=(0, 0), rotation=0)
            self.robot_circle.add_attr(self.robot_transform)
            self.robot_line.add_attr(self.robot_transform)
            self.viewer.add_geom(self.robot_circle)
            self.viewer.add_geom(self.robot_line)

            # Draw exit location
            self.exit = rendering.make_circle(self.EXIT_RADIUS, filled=False)
            xfrm = Transform(translation=self._reshape(self.exit_location))
            self.exit.add_attr(xfrm)
            self.viewer.add_geom(self.exit)

            # Set up for showing trajectory
            self.trajectory = []

        # Draw walls
        for wall in self.WALLS:
            self._draw_line(wall, (0, 0, 0))

        # Show sensors if indicated
        if show_sensors:

            from pyglet.text import Label

            # Show rangefinders as line segments
            for i, line in enumerate(self.rangefinder_lines):
                self._draw_line(line, (0, 1, 0) if i == 2 else (1, 0, 0))

            # Show radar as cardinal direction
            x, y = self._reshape(self.location)
            label = Label(font_size=12,
                          x=x-5,
                          y=y,
                          anchor_x='left',
                          anchor_y='center',
                          color=(0, 0, 0, 255))
            label.text = 'FRBL'[np.argmax(self.radar)]
            self.viewer.add_onetime(_DrawText(label))

        else:

            # Draw robot
            self.robot_transform.set_translation(*self._reshape(self.location))
            self.robot_transform.set_rotation(-np.radians(self.heading))

        # Show trajectory if indicated
        if show_trajectory:
            self.trajectory.append(self.location)
            for i in range(len(self.trajectory)-1):
                self._draw_line((self.trajectory[i],
                                self.trajectory[i+1]),
                                (0, 0, 1))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):

        return

    def _draw_line(self, line, color):

        self.viewer.draw_line(self._reshape(line[0]),
                              self._reshape(line[1]),
                              color=color)

    def _update_velocity(self, v, a):

        return np.clip(v + (a - 0.5), -self.MAX_SPEED, +self.MAX_SPEED)

    def _reshape(self, point):
        '''
        Allows us to keep original maze coordinates while presenting it in a
        larger, upside-down format.
        '''
        return 2*point[0], (self.maze_size[1]-2*point[1])

    def _collision(self, loc):

        return any([distance_point_to_line(loc, wall) <
                   self.ROBOT_RADIUS for wall in self.walls])

    def _update_rangefinders(self):

        # Iterate through each sensor and find distance to maze lines with
        # agent's rangefinder sensors
        for i, angle in enumerate(self.RANGEFINDER_ANGLES):

            rad = np.radians(angle)

            # Project a point from the robot's location outwards
            projection_point = (
                    self.location[0] + np.cos(rad) * self.RANGEFINDER_RANGE,
                    self.location[1] + np.sin(rad) * self.RANGEFINDER_RANGE)

            # Rotate the projection point by the robot's heading
            projection_point = rotate(projection_point,
                                      self.location,
                                      self.heading)

            # create a line segment from the robot's location to projected
            projection_line = [tuple(self.location), projection_point]

            # Store line for rendering
            self.rangefinder_lines[i] = projection_line

            # Set range to max by default
            min_range = self.RANGEFINDER_RANGE

            # Now test against the environment to see if we hit anything
            for wall in self.walls:

                wallpoint = intersection(wall, projection_line)

                if wallpoint is not None:

                    # Store intersection for rendering
                    self.rangefinder_lines[i][1] = wallpoint

                    # If so, then update the range to the distance
                    found_range = distance_point_to_point(wallpoint,
                                                          self.location)

                    # We want the closest intersection
                    min_range = min(found_range, min_range)

            self.rangefinders[i] = min_range

    def _update_radar(self):

        target = np.array(self.EXIT_LOCATION)

        # Rotate goal with respect to heading of agent to compensate agent's
        # heading angle relative to zero heading angle
        target = rotate(target, self.location, -self.heading)

        # Translate with respect to location of agent to compensate agent's
        # position relative to (0,0)
        target -= self.location

        # What angle is the vector between target & agent (agent is placed into
        # (0,0) with zero heading angle due to the affine transforms above)?
        angle = arctan_degrees(target)

        # Fire the appropriate radar sensor
        for i, _angle in enumerate(self.RADAR_ANGLES):
            self.radar[i] = 0
            lo = self.RADAR_ANGLES[i]
            hi = lo + 90
            if ((angle >= lo and angle < hi) or
               (angle + 360 >= lo and angle + 360 < hi)):
                self.radar[i] = 1


class MazeMedium(Maze):

    INITIAL_LOCATION = 30, 22
    INITIAL_HEADING = 0

    EXIT_LOCATION = 270, 100

    WALLS = (
            ((5,    5),  (295, 5)),
            ((295,  5),  (295, 135)),
            ((295, 135), (5, 135)),
            ((5,   135), (5, 5)),
            ((241, 135), (58, 65)),
            ((114, 5),   (73, 42)),
            ((130, 91),  (107, 46)),
            ((196, 5),   (139, 51)),
            ((219, 125), (182, 63)),
            ((267, 5),   (214, 63)),
            ((271, 135), (237, 88))
            )

    def __init__(self):

        Maze.__init__(self,
                      self.INITIAL_LOCATION,
                      self.INITIAL_HEADING,
                      self.EXIT_LOCATION,
                      self.WALLS)


class MazeHard(Maze):

    INITIAL_LOCATION = 36, 184
    INITIAL_HEADING = 0

    EXIT_LOCATION = 31, 20

    WALLS = (
            ((5, 5), (5, 200)),
            ((5, 200), (200, 200)),
            ((200, 200), (200, 5)),
            ((200, 5), (5, 5)),
            ((5, 49), (57, 53)),
            ((56, 54), (56, 157)),
            ((57, 106), (158, 162)),
            ((77, 200), (108, 164)),
            ((5, 80), (33, 121)),
            ((200, 146), (87, 91)),
            ((56, 55), (133, 30)),
            )

    def __init__(self):

        Maze.__init__(self,
                      self.INITIAL_LOCATION,
                      self.INITIAL_HEADING,
                      self.EXIT_LOCATION,
                      self.WALLS)


def maze_demo(env):
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
    parser.add_argument('--traj', dest='show_trajectory',
                        action='store_true', help='Show trajectory')
    args = parser.parse_args()

    env.max_steps = args.steps
    env.seed(args.seed)
    np.random.seed(args.seed)

    state = env.reset()

    for k in range(args.steps):

        action = np.random.random(2)

        state, reward, _, info = env.step(action)

        frame = env.render(mode='rgb_array',
                           show_sensors=args.show_sensors,
                           show_trajectory=(args.show_trajectory))
        sleep(1./env.FRAMES_PER_SECOND)

        if frame is None:
            break

        last = (k == args.steps - 1)

        if k % 20 == 0 or last:
            print('step  %05d/%05d' % (k, env.max_steps), end='')
            if last:
                print('  reward = %f' % reward, end='')
                if last and args.use_novelty:
                    print('  behavior =', info['behavior'], end='')
            print()

    sleep(1)
    env.close()
