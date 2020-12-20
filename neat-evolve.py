#!/usr/bin/env python3
'''
NEAT evolver script for OpenAI Gym environemnts

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from neat_gym import _GymConfig, _evolve

_evolve(_GymConfig.make_config)
