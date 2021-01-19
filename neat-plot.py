#!/usr/bin/env python3
'''
Script for plotting results of evolution run

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import argparse
from numpy import genfromtxt
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('csvfile', metavar='CSVFILE', help='input .csv file')
args = parser.parse_args()


data = genfromtxt(args.csvfile, delimiter=',', skip_header=1)

g = data[:, 0]
mnfit = data[:, 1]

plt.plot(g, mnfit)
plt.plot(g, data[:, 3])
plt.errorbar(g, mnfit, data[:, 2], linestyle='None')

plt.xlabel('Generation')

plt.legend(['Mean Fitness', 'Max Fitness'])

plt.show()
