'''
Copyright (C) 2020 Simon D. Levy

MIT License
'''

from gym.envs.registration import register

register(
    id='Medium-v0',
    entry_point='gym_nsmaze.envs:Medium',
    max_episode_steps=400
)

register(
    id='Hard-v0',
    entry_point='gym_nsmaze.envs:Hard',
    max_episode_steps=400
)
