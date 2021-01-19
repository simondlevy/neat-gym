#!/usr/bin/env python3
'''
Script for plotting results of evolution run

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import argparse
from numpy import genfromtxt

parser = argparse.ArgumentParser()
parser.add_argument('csvfile', metavar='CSVFILE', help='input .csv file')
args = parser.parse_args()


data = genfromtxt(args.csvfile, delimiter=',', skip_header=1)

print(data)
