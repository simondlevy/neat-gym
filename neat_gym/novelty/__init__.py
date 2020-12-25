#!/usr/bin/env python3
'''
Novelty Search in Python

Copyright (C) 2020 Simon D. Levy

MIT License
'''


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

    Copyright (C) 2020 Simon D. Levy

    MIT License
    '''

    def __init__(self, k, threshold, limit, ndims):
        '''
        Creates an object supporting Novelty Search.
        @param k k for k-nearest-neighbors
        @param threshold threshold for how novel an example has to be before it will be added the archive
        @param limit maximum size of the archive.
        @param ndims dimensionality of archive elements
        '''
        self.k = k
        self.threshold = threshold
        self.limit = limit

        # Archive implemented as a circular buffer
        self.archive = np.zeros((limit,ndims))
        self.count = 0

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

        if self.count < self.limit or s > self.threshold:
            self.archive[self.count%self.limit] = np.array(p)
            self.count += 1

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
