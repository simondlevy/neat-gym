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
    plt.errorbar(g, mn, sd, linestyle='None')

    plt.xlabel('Generation')
    plt.legend(['Mean ' + lbl, 'Max ' + lbl])


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('csvfile', metavar='CSVFILE', help='input .csv file')
    args = parser.parse_args()

    try:
        data = np.genfromtxt(args.csvfile, delimiter=',', skip_header=1)
    except Exception:
        print('Unable to open file %s' % args.csvfile)
        exit(1)

    # Data from Novelty Search
    if data.shape[1] > 4:

        plt.subplot(2, 1, 2)
        plot(data, 4, 'b', 'k', 'Novelty')
        plt.subplot(2, 1, 1)

    plot(data, 1, 'g', 'm', 'Fitness')

    plt.title(args.csvfile)
    plt.show()


main()
