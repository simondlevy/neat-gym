#!/usr/bin/env python3
'''
Novelty Search test based on
    https://www.cs.swarthmore.edu/~meeden/cs81/s12/lab4.php

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import pickle
import numpy as np
from neat_gym.novelty import Novelty


def test(nov):
    #    Point (0.5, 0.5) should have a low sparseness since it is at the
    #    center of the range of random points you generated.
    #
    #    Point (1, 1) should have a higher sparseness since it is at the
    #    boundary of random points you generated.
    #
    #    Point (2, 2) should have an even higher sparseness since it is outside
    #    the range of random points you generated.
    #
    #    Point (5, 5) should have the highest sparseness since it is far
    #    outside the range of random points you generated.

    print('point      sparseness\n-----      ----------')
    for p in [(0.5, 0.5), (1., 1.), (2., 2.), (5., 5.)]:
        print(p, nov._sparseness(p))


def main(seed=None):

    # Seed the random-number generator for reproducibility.
    np.random.seed(seed)

    # Create an instance of your Novelty class with a k of 10, a threshold of
    # 0.3, a limit of 100, and  a dimensionality of 2.
    nov = Novelty(10, 0.3, 10, 2)

    # Use a for loop to generate 10 random 2d points, where each value is in
    # the range [0.0, 0.3], and add them to the archive.
    for _ in range(10):
        nov.add(0.3 * np.random.random(2))

    # Use another for loop to generate 10 random 2d points, where each value is
    # in the range [0.0, 1.0], and add them to the archive.
    for _ in range(10):
        nov.add(np.random.random(2))

    # At the end of the main program print the sparseness of several points:
    test(nov)

    # Save the Novelty object to a file called test.dat.
    pickle.dump(nov, open('test.dat', 'wb'))

    # Load the saved object and run the sparseness test again
    nov = pickle.load(open('test.dat', 'rb'))
    test(nov)


if __name__ == '__main__':
    main()
