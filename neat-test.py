#!/usr/bin/env python3
'''
Test script for using NEAT with OpenAI Gym environments

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import gym
from neat_gym import read_file, eval_net

# Load genome and configuration from pickled file
net, env_name, record_dir, nodisplay = read_file(allow_record=True)

# Run the network on the environment
print('Reward = %6.6f.' % eval_net(
                            net,
                            gym.make(env_name),
                            render=(not nodisplay),
                            record_dir=record_dir))
