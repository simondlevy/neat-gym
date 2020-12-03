#!/usr/bin/env python3
import gym
import neat 
from pureples.shared.gym_runner import run_neat

config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
        neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
        'config_neat_mountain_car')

env = gym.make('MountainCar-v0')

run_neat(1, env, 8, config, max_trials=0)

