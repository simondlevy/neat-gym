#!/usr/bin/env python3
import gym
import neat 
import numpy as np

def run_neat(gens, env, max_steps, config, max_trials=100, output=True):

    def eval_fitness(genomes, config):

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

    pop = neat.population.Population(config)

    pop.run(eval_fitness, gens)

config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
        neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
        'config_neat_mountain_car')

env = gym.make('MountainCar-v0')

run_neat(1, env, 8, config, max_trials=0)

