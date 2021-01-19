#!/usr/bin/env python3
'''
Script for plotting results of evolution run

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('csvfile', metavar='CSVFILE', help='input .csv file')
args = parser.parse_args()

try:
    data = np.genfromtxt(args.csvfile, delimiter=',', skip_header=1)
except Exception:
    print('Unable to open file %s' % args.csvfile)
    exit(1)

print(data)
exit(0)

g = data[:, 0]
mnfit = data[:, 1]
sdfit = data[:, 2]
mxfit = data[:, 3]
plt.plot(g, mnfit)
plt.plot(g, mxfit)
plt.errorbar(g, mnfit, sdfit, linestyle='None')

plt.xlabel('Generation')
plt.title(args.csvfile)
plt.legend(['Mean Fitness', 'Max Fitness'])

plt.show()
