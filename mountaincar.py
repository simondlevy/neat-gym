#!/usr/bin/env python3
import gym
import neat 
import numpy as np

def ini_pop(state, stats, config, output):
    pop = neat.population.Population(config, state)
    if output:
        pop.add_reporter(neat.reporting.StdOutReporter(True))
    pop.add_reporter(stats)
    return pop


def run_neat(gens, env, max_steps, config, max_trials=100, output=True):
    trials = 1

    def eval_fitness(genomes, config):

        for idx, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, config)

            fitnesses = []

            for i in range(trials):
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

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output)
    pop.run(eval_fitness, gens)

    stats_ten = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, output)
    trials = 10
    winner_ten = pop.run(eval_fitness, gens)

    if max_trials is 0:
        return winner_ten, (stats_one, stats_ten)

    stats_hundred = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_hundred, config, output)
    trials = max_trials
    winner_hundred = pop.run(eval_fitness, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)

config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
        neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
        'config_neat_mountain_car')

env = gym.make('MountainCar-v0')

run_neat(1, env, 8, config, max_trials=0)

