#!/usr/bin/env python3
import gym
import neat 
import numpy as np

def eval_genome(genome, config, env):

    max_steps = 200
    evals = 2

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for i in range(evals):

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
    
    genome.fitness = np.array(fitnesses).mean()

def eval_genomes(genomes, config):

    env = gym.make('MountainCar-v0')

    for _, g in genomes:

        eval_genome(g, config, env)

def main():

    config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
            'config_neat_mountain_car')

    pop = neat.population.Population(config)

    pop.run(eval_genomes, 1)

if __name__ == '__main__':
    main()

