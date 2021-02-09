#!/usr/bin/env python3
'''
Test maze methods

Adapted from https://github.com/yaricom/goNEAT_NS/blob/master/experiments/maze/environment_test.go

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import numpy as np

from envs import Maze

def error(msg):
    print(msg)
    exit(1)

def TestPoint_Angle():

    for point, expected in zip(
            ((+1,0), (0,+1), (-1,0), (0,-1), (+1,+1), (-1,+1), (-1,-1), (+1,-1)),
            (   0,     90,     180,    270,    45,      135,     225,     315)):

        angle = Maze._angle(point)

        if angle != expected:
            error('Wrong angle found: %f, expected: %f', angle, expected)

    print('Maze._angle passed tests.')


def TestPoint_Rotate():

    p = Maze._rotate((2,1), (1,1), 90)

    if p != (1,2):
        error('Wrong position after rotation: expected (1,2); got ' + str(p))
        exit(1)

    p = Maze._rotate(p, (1,1), 180)

    if 1 - p[0] > 1e-8 or p[1] != 0:
        error('Wrong position after rotation: expected (0,0); got ' + str(p))
        exit(1)

    print('Maze._rotate passed tests.')

def TestPoint_Distance():

    p = 2, 1
    q = 5, 1

    d = Maze._distance_point_to_point(p, q)
    if d != 3:
        error('Wrong distance', d)

    q = 5, 3
    d = Maze._distance_point_to_point(p, q)
    if d != np.sqrt(13):
        error('Wrong distance', d)

    print('Maze._distance passed tests.')

def TestLine_Intersection():

    l1 = ((1, 1), (5, 5))
    l2 = ((1, 5), (5, 1))

    # test intersection

    p = Maze._intersection(l1, l2)

    if p != (3,3):
        error('Wrong intersection found: should be (3,3); got ' + str(p))

    # test parallel

    l3 = ((2, 1), (6, 1))

    p = Maze._intersection(l1, l3)

    if p is not None:
        error('Wrong intersection point found: should be (0,0); got ' + str(p))
   
    # test no intersection by coordinates

    l4 = ((4, 4), (6, 1))

    p = Maze._intersection(l1, l4)

    if p is not None:
        error('The lines must not intesect')

    print('Maze._intersection passed tests.')

def main():

    TestPoint_Angle()
    TestPoint_Rotate()
    TestPoint_Distance()
    TestLine_Intersection()

if __name__ == '__main__':

    main()




















