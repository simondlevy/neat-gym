'''
Copyright (C) 2021 Simon D. Levy

MIT License
'''

from gym.envs.registration import register

register(
    id='Gather-v0',
    entry_point='gym_gather.envs:FoodGatherConcentric'
)
