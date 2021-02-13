#!/usr/bin/env python3
'''
Script for plotting results of evolution run

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot(data, usetime, beg, s1, s2, lbl):

    if len(data.shape) < 2:
        print('Just one point: Mean = %+3.3f  Std = %+3.3f  Max = %+3.3f' %
              (data[1], data[2], data[3]))
        exit(0)

    g = data[:, 0]
    t = data[:, 1]

    mn = data[:, beg]
    sd = data[:, beg+1]
    mx = data[:, beg+2]

    x = t if usetime else g

    plt.plot(x, mn, s1)
    plt.plot(x, mx, s2)
    # plt.errorbar(g, mn, sd, linestyle='None')
    plt.plot(x, mn + sd, 'g--')
    plt.plot(x, mn - sd, 'g--')

    plt.xlabel('Time(sec)' if usetime else 'Generation')
    plt.legend(['Mean ' + lbl, 'Max ' + lbl, '+/-1 StdDev'])


def plot_novelty(data, usetime):

    plot(data, usetime, 5, 'b', 'k', 'Novelty')


def plot_fitness(data, usetime):

    plot(data, usetime, 2, 'r', 'm', 'Fitness')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('csvfile', metavar='CSVFILE', help='input .csv file')
    parser.add_argument('--time', dest='time', action='store_true',
                        help='Plot against time instead of episode')
    parser.add_argument('--split', action='store_true',
                        help='Use separate figure for novelty')
    args = parser.parse_args()

    try:
        data = np.genfromtxt(args.csvfile, delimiter=',', skip_header=1)
    except Exception:
        print('Unable to open file %s' % args.csvfile)
        exit(1)

    # Data from Novelty Search
    if len(data.shape) > 1 and data.shape[1] > 5:

        if args.split:
            plot_fitness(data, args.time)
            plt.title(args.csvfile)
            plt.figure()
            plot_novelty(data, args.time)
            plt.title(args.csvfile)

        else:
            plt.subplot(2, 1, 1)
            plot_fitness(data, args.time)
            plt.title(args.csvfile)
            plt.subplot(2, 1, 2)
            plot_novelty(data, args.time)

    else:

        if args.split:
            print('No novelty data')

        plot_fitness(data, args.time)
        plt.title(args.csvfile)

    plt.show()


main()
