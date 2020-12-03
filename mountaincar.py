#!/usr/bin/env python3
import gym
import neat 
import numpy as np

def eval_fitness(genomes, config):

    env = gym.make('MountainCar-v0')
    max_steps = 200

    for idx, g in genomes:

        net = neat.nn.FeedForwardNetwork.create(g, config)

        fitnesses = []

        for i in range(1):
            ob = env.reset()

            total_reward = 0

            for j in range(max_steps):
                o = net.activate(ob)
                action = np.argmax(o)
                ob, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break
            fitnesses.append(total_reward)
        
        g.fitness = np.array(fitnesses).mean()

config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
        neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
        'config_neat_mountain_car')

pop = neat.population.Population(config)

pop.run(eval_fitness, 1)

