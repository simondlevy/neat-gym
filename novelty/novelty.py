#!/usr/bin/env python3

import numpy as np

class Novelty(object):
    '''
    Implementation of Novelty Search as described in

        @article{article,
        author = {Lehman, Joel and Stanley, Kenneth},
        year = {2011},
        month = {06},
        pages = {189-223},
        title = {Abandoning Objectives: Evolution Through the Search for Novelty Alone},
        volume = {19},
        journal = {Evolutionary computation},
        doi = {10.1162/EVCO_a_00025}
        }

    API based on https://www.cs.swarthmore.edu/~meeden/cs81/s12/lab4.php

    Copyright (C) 2020 Simon D. Levy

    MIT License
    '''

    def __init__(self, k, threshold, limit):
        '''
        Creates an object supporting Novelty Search.
        @param k k for k-nearest-neighbors
        @param threshold threshold for how novel an example has to be before it will be added the archive
        @param limit maximum size of the archive.
        '''
        self.k = k
        self.threshold = threshold
        self.limit = limit
        self.archive = []

    def __str__(self):

        return 'Novelty k = %d  threshold = %f  limit = %d' % (self.k, self.threshold, self.limit)

    def add(self, p):
        '''
        If the size of the archive is less than the limit, adds the point p.
        Otherwise, when the archive is full, checks whether sparseness of p is
        greater than the threshold. If so, adds p and removes oldest point in
        archive.
        @param p the point
        @return sparseness of point in archive
        '''

        s = self._sparseness(p)
       
        if len(self.archive) < self.limit:
            self.archive.append(p)

        elif s < self.threshold:
            self.archive = self.archive[1:] + [p]

        return s

    def saveArchive(self, filename):
        '''
        Saves the entire archive to the given filename by writing each archived
        point on a single line in the file, with spaces separating each value.
        @parm filename
        '''
        with open(filename, 'w') as f:
            for p in self.archive:
                for x in p:
                    f.write('%f ' % x)
                f.write('\n')

    def _distance(self, p1, p2):
        '''
        Returns the L2 distance between points p1 and p2 which are assumed to be
        lists or tuples of equal length. 
        '''

        assert(len(p1) == len(p2))

        return np.sqrt(np.sum((np.array(p1)-np.array(p2))**2))

    def _distFromkNearest(self, p):
        '''
        Returns the distance of a point p from its k-nearest neighbors in the
        archive.
        '''

        # XXX
        # The simplest, though very inefficient, way to implement this
        # is to calculate the distance of p from every point in the archive, sort
        # these distances, and then sum up and return the first k (which will be
        # the closest).  

        return np.sum(np.sort([self._distance(p, q) for q in self.archive])[:self.k])

    def _sparseness(self, p):
        '''
        Returns the sparseness of the given point p as defined by equation 1 on
        page 13 of Lehman & Stanley 2011. Recall that sparseness is a measure
        of how unique this point is relative to the archive of saved examples.
        Use the method distFromkNearest as a helper in calculating this value.  
        '''

        return 1./self.k * self._distFromkNearest(p)
        
def simple_test(seed=None):

    # Seed the random-number generator for reproducibility.
    np.random.seed(seed)

    # Create an instance of your Novelty class with a k of 10, a threshold of
    # 0.3, and a limit of 100.
    nov = Novelty(10, 0.3, 100)

    # Use a for loop to generate 1000 random 2d points, where each value is in the
    # range [0.0, 0.3], and add them to the archive.
    for _ in range(1000):
        nov.add(0.3 * np.random.random(2))

    # Use another for loop to generate 1000 random 2d points, where each value is in
    # the range [0.0, 1.0], and add them to the archive.
    for _ in range(1000):
        nov.add(np.random.random(2))

    # After both loops, save the archive to a file called test.dat.
    nov.saveArchive('test.dat')

    # At the end of the main program print the sparseness of several points: 
    #
    #    Point (0.5, 0.5) should have a low sparseness since it is at the center of the range
    #    of random points you generated.
    #
    #    Point (1, 1) should have a higher sparseness since it is at the boundary of random points you generated.
    #
    #    Point (2, 2) should have an even higher sparseness since it is outside the range of random points you generated.
    #
    #    Point (5, 5) should have the highest sparseness since it is far
    #    outside the range of random points you generated.
    print('point      sparseness\n-----      ----------')
    for p in [(0.5,0.5), (1.,1.), (2.,2.), (5.,5.)]:
        print(p, nov._sparseness(p))

def xor_test(seed=None):

    # Seed the random-number generator for reproducibility.
    np.random.seed(seed)

    # Create an instance of your Novelty class with a k of 10, a threshold of
    # 0.3, and a limit of 150.
    nov = Novelty(10, 0.3, 150)

# Tests
if __name__ == '__main__':
    simple_test()
    xor_test()
