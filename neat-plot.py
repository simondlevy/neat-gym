#!/usr/bin/env python3
'''
Script for plotting results of evolution run

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot(data, beg, s1, s2, lbl):

    g = data[:, 0]

    mn = data[:, beg]
    sd = data[:, beg+1]
    mx = data[:, beg+2]

    plt.plot(g, mn, s1)
    plt.plot(g, mx, s2)
    # plt.errorbar(g, mn, sd, linestyle='None')
    plt.plot(g, mn + sd, 'g--')
    plt.plot(g, mn - sd, 'g--')

    plt.xlabel('Generation')
    plt.legend(['Mean ' + lbl, 'Max ' + lbl])


def plot_novelty(data):

    plot(data, 4, 'b', 'k', 'Novelty')


def plot_fitness(data):

    plot(data, 1, 'r', 'm', 'Fitness')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('csvfile', metavar='CSVFILE', help='input .csv file')
    parser.add_argument('--split', action='store_true',
                        help='Make two separate plots')
    args = parser.parse_args()

    try:
        data = np.genfromtxt(args.csvfile, delimiter=',', skip_header=1)
    except Exception:
        print('Unable to open file %s' % args.csvfile)
        exit(1)

    # Data from Novelty Search
    if data.shape[1] > 4:

        if args.split:
            plot_fitness(data)
            plt.title(args.csvfile)
            plt.figure()
            plot_novelty(data)
            plt.title(args.csvfile)

        else:
            plt.subplot(2, 1, 1)
            plot_fitness(data)
            plt.title(args.csvfile)
            plt.subplot(2, 1, 2)
            plot_novelty(data)

    else:

        if args.split:
            print('No novelty data')

        plot_fitness(data)

    plt.show()


main()
