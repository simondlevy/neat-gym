#!/usr/bin/env python3
import gym
import neat 
import numpy as np

class GymConfig(neat.config.Config):

    def __init__(self, genome_type, reproduction_type, species_set_type, stagnation_type, filename, env_name):

        neat.config.Config.__init__(self, genome_type, reproduction_type, species_set_type, stagnation_type, filename)

        self.env = gym.make(env_name)

def eval_genome(genome, config):

    max_steps = 200
    evals = 2

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for i in range(evals):

        ob = config.env.reset()

        total_reward = 0

        for j in range(max_steps):
            o = net.activate(ob)
            action = np.argmax(o)
            ob, reward, done, info = config.env.step(action)
            total_reward += reward
            if done:
                break
        fitnesses.append(total_reward)
    
    genome.fitness = np.array(fitnesses).mean()

def eval_genomes(genomes, config):

    for _, g in genomes:

        eval_genome(g, config)

def main():

    config = GymConfig(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
            'config_neat_mountain_car', 'MountainCar-v0')

    pop = neat.population.Population(config)

    pop.run(eval_genomes, 1)

if __name__ == '__main__':
    main()

