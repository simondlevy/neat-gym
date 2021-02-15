'''
Copyright (C) 2021 Simon D. Levy

MIT License
'''

from gym.envs.registration import register

register(
    id='Maze-v0',
    entry_point='neat_gym_tests.envs:MazeMedium',
    max_episode_steps=400
)

register(
    id='Maze-v1',
    entry_point='neat_gym_tests.envs:MazeHard',
    max_episode_steps=400
)
