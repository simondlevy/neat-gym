#!/usr/bin/env python3
'''
Novelty Search in Python

Copyright (C) 2020 Simon D. Levy

MIT License
'''


import numpy as np
from rtree import index

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
        @param k for k-nearest-neighbors
        @param threshold how novel an example has to be before it will be added the archive
        @param limit maximum size of the archive.
        @param ndims dimensionality of archive elements
        '''
        self.k = k
        self.threshold = threshold
        self.limit = limit
        self.ndims = ndims

        # Archive implemented as a circular buffer
        self.archive = np.zeros((limit,ndims))
        self.count = 0

        # Create an Rtree Index for inserting the points
        p = index.Property()
        p.dimension = ndims
        self.knn = index.Index(properties=p, interleaved=False)

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

        # Start with zero as sparseness
        s = 0
 
        # Below limit, fill archive and ignore actual sparseness
        if self.count < self.limit:

            self.archive[self.count] = np.array(p)

            # Insert new point in kNN.  With interleaved=False, the order of
            # input and output is: (xmin, xmax, ymin, ymax, zmin, zmax, # ...)
            self.knn.insert(self.count, Novelty._expand_point(p))

            self.count += 1

        else: 

            # Compute sparseness of new point
            s = self._sparseness(p)
            
            # If sparseness excedes threshold, ...
            if s > self.threshold:

                # Implement a circular buffer
                idx = self.count % self.limit

                # Remove old point from kNN
                self.knn.delete(idx, Novelty._expand_point(self.archive[idx]))

                # Insert new point in kNN.  With interleaved=False, the order of
                # input and output is: (xmin, xmax, ymin, ymax, zmin, zmax, # ...)
                self.knn.insert(idx, Novelty._expand_point(p))

                # Store new point in archive
                self.archive[idx] = np.array(p)

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

    def _sparseness(self, p):
        '''
        Returns the sparseness of the given point p as defined by equation 1 on
        page 13 of Lehman & Stanley 2011. Recall that sparseness is a measure
        of how unique this point is relative to the archive of saved examples.
        '''

        nbrs_old = list(np.argsort([Novelty._distance(p, q) for q in self.archive])[:self.k])
        nbrs_new = list(self.knn.nearest(p, self.k))

        if np.any(nbrs_old != nbrs_new):
            print('p: ', p)
            print('nbrs_old: ', sorted(nbrs_old))
            print('nbrs_new: ', sorted(nbrs_new))
            exit(0)

        return 1./self.k * np.sqrt(np.sum(np.sum((self.archive[nbrs_old,:] - p)**2, axis=1)))

    @staticmethod
    def _distance(p1, p2):
        '''
        Returns the L2 distance between points p1 and p2 which are assumed to be
        lists or tuples of equal length. 
        '''
        return np.sqrt(np.sum((np.array(p1)-np.array(p2))**2))

    @staticmethod
    def _expand_point(pt):
        return tuple(item for sublist in [(x,x) for x in pt] for item in sublist)
