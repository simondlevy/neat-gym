#!/usr/bin/env python3
'''
HyperNEAT evolver script for OpenAI Gym environemnts

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from neat_gym import _GymHyperConfig, _evolve

_evolve(_GymHyperConfig.make_config)
