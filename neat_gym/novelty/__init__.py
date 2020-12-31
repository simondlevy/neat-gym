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
            self.knn.insert(self.count, tuple(item for sublist in [(x,x) for x in p] for item in sublist))

            self.count += 1

        else: 

            # Compute sparseness of new point
            s = self._sparseness(p)
            
            # If sparseness excedes threshold, ...
            if s > self.threshold:

                # Implement a circular buffer
                idx = self.count % self.limit

                # Remove old point from kNN
                self.knn.delete(idx, self.archive[idx])

                # Insert new point in kNN.  With interleaved=False, the order of
                # input and output is: (xmin, xmax, ymin, ymax, zmin, zmax, # ...)
                self.knn.insert(idx, tuple(item for sublist in [(x,x) for x in p] for item in sublist))

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

        # Get k nearest neighbors
        nbrs = list(self.knn.nearest(p, self.k))

        # Compute the distance of the point from these neighbors
        dst = 0
        for j in range(self.ndims):
            dst += np.sum((p[j] - self.archive[nbrs,j])**2)

        # Apply the equation to get the point's sparseness
        return 1./self.k * np.sqrt(dst)
