# This file is part of atooms
# Copyright 2010-2018, Daniele Coslovich

"""Fourier-space post processing code."""

import math
import logging
import random
from collections import defaultdict

import numpy

from .helpers import linear_grid
from .correlation import Correlation

__all__ = ['expo_sphere', 'expo_sphere_safe', 'FourierSpaceCorrelation']

_log = logging.getLogger(__name__)


def expo_sphere(k0, kmax, pos):

    """Returns the exponentials of the input positions for each k."""

    # Technical note: we use ellipsis, so that we can pass either a
    # single sample or multiple samples without having to add a
    # trivial extra dimension to input array
    im = numpy.complex(0.0, 1.0)
    # The integer grid must be the same as the one set in kgrid,
    # otherwise there is an offset the problem is that integer
    # negative indexing is impossible in python and rounding or
    # truncating kmax can slightly offset the grid

    # We pick up the smallest k0 to compute the integer grid
    # This leaves many unused vectors in the other directions, which
    # could be dropped using different nkmax for x, y, z
    nk_max = 1 + int(kmax / min(k0))
    # The shape of expo is nframes, N, ndim, 2*nk+1
    expo = numpy.ndarray((len(pos), ) + pos[0].shape + (2*nk_max+1, ), numpy.complex)
    expo[..., nk_max] = numpy.complex(1.0, 0.0)
    # First fill positive k
    for j in range(pos[0].shape[-1]):
        expo[..., j, nk_max+1] = numpy.exp(im * k0[j] * pos[..., j])
        expo[..., j, nk_max-1] = expo[..., j, nk_max+1].conjugate()
        for i in range(2, nk_max):
            expo[..., j, nk_max+i] = expo[..., j, nk_max+i-1] * expo[..., j, nk_max+1]
    # Then take complex conj for negative ones
    for i in range(2, nk_max+1):
        expo[..., nk_max+i] = expo[..., nk_max+i-1] * expo[..., nk_max+1]
        expo[..., nk_max-i] = expo[..., nk_max+i].conjugate()

    return expo


def expo_sphere_safe(k0, kmax, pos):
    """
    Returns the exponentials of the input positions for each k.
    It does not use ellipsis.
    """
    im = numpy.complex(0.0, 1.0)
    ndims = pos.shape[-1]
    nk_max = 1 + int(kmax / min(k0))
    expo = numpy.ndarray(pos.shape + (2*nk_max+1, ), numpy.complex)
    expo[:, :, :, nk_max] = numpy.complex(1.0, 0.0)

    for j in range(ndims):
        expo[:, :, j, nk_max+1] = numpy.exp(im*k0[j]*pos[:, :, j])
        expo[:, :, j, nk_max-1] = expo[:, :, j, nk_max+1].conjugate()
        for i in range(2, nk_max):
            expo[:, :, j, nk_max+i] = expo[:, :, j, nk_max+i-1] * expo[:, :, j, nk_max+1]

    for i in range(2, nk_max+1):
        expo[:, :, :, nk_max+i] = expo[:, :, :, nk_max+i-1] * expo[:, :, :, nk_max+1]
        expo[:, :, :, nk_max-i] = expo[:, :, :, nk_max+i].conjugate()

    return expo


def _k_norm(ik, k0, offset):
    k_shift = k0 * (numpy.array(ik) - offset)
    k_sq = numpy.dot(k_shift, k_shift)
    return math.sqrt(k_sq)

def _sphere(kmax):
    ikvec = numpy.ndarray(3, dtype=numpy.int)
    for ix in range(-kmax, kmax+1):
        for iy in range(-kmax, kmax+1):
            for iz in range(-kmax, kmax+1):
                ikvec[0] = ix
                ikvec[1] = iy
                ikvec[2] = iz
                yield ikvec
                
def _disk(kmax):
    ikvec = numpy.ndarray(2, dtype=numpy.int)
    for ix in range(-kmax, kmax+1):
        for iy in range(-kmax, kmax+1):
            ikvec[0] = ix
            ikvec[1] = iy
            yield ikvec
                        

class FourierSpaceCorrelation(Correlation):

    """
    Base class for Fourier space correlation functions.

    The correlation function is computed for each of the scalar values
    k_i of the provided `kgrid`. If the latter is `None`, the grid is
    built using `ksamples` entries linearly spaced between `kmin` and
    `kmax`.

    For each sample k_i in `kgrid`, the correlation function is
    computed over at most `nk` wave-vectors (k_x, k_y, k_z) such that
    their norm (k_x^2+k_y^2+k_z^2)^{1/2} lies within `dk` of the
    prescribed value k_i.

    See the doc of `Correlation` for information about the rest of the
    instance variables.
    """

    def __init__(self, trajectory, grid, norigins=None, nk=8, dk=0.1,
                 kmin=-1, kmax=10, ksamples=20, fix_cm=False, normalize=True):
        super(FourierSpaceCorrelation, self).__init__(trajectory,
                                                      grid, norigins=norigins, fix_cm=fix_cm)
        # Some additional variables. k0 = smallest wave vectors
        # compatible with the boundary conditions
        # TODO: document the additional data structures used to store k vectors
        # TODO: streamline k vectors data structures
        self.normalize = normalize
        self.nk = nk
        self.dk = dk
        self.kmin = kmin
        self.kmax = kmax
        self.ksamples = ksamples
        self.kgrid = []
        self.k0 = []
        self.kvector = {}
        self.selection = []
        self._kbin_max = 0

    def compute(self):
        # We subclass compute to define k grid at compute time
        # Find k-norms grid and store it a self.kgrid (the norms are sorted)
        variables = self.short_name.split('(')[1][:-1]
        variables = variables.split(',')
        if len(variables) > 1:
            self.kgrid = self.grid[variables.index('k')]
        else:
            self.kgrid = self.grid

        # _setup() assumes sorted kgrid.
        self.kgrid.sort()
        # Setup grid once. If cell changes we'll call it again
        self._setup()

        # Pick up a random, unique set of nk vectors out of the available ones
        # without exceeding maximum number of vectors in shell nkmax
        self.kgrid, self.selection = self._decimate_k()
        # We redefine the grid because of slight differences on the
        # average k norms appear after decimation.
        self.kgrid = self._actual_k_grid()
        # We must fix the keys: just pop them to the their new positions
        # We sort both of them (better check len's)
        for k, kv in zip(sorted(self.kgrid), sorted(self.kvector)):
            self.kvector[k] = self.kvector.pop(kv)

        # Now compute
        super(FourierSpaceCorrelation, self).compute()

    def _setup(self, sample=0):
        self.k0 = 2*math.pi/self.trajectory[sample].cell.side
        # If grid is not provided, setup a linear grid from kmin,kmax,ksamples data
        # TODO: This shouldnt be allowed with fluctuating cells
        # Or we should fix the smallest k to some average of smallest k per sample
        if self.kgrid is None:
            if self.kmin > 0:
                self.kgrid = linear_grid(self.kmin, self.kmax, self.ksamples)
            else:
                self.kgrid = linear_grid(min(self.k0), self.kmax, self.ksamples)
        else:
            # If the first wave-vector is negative we replace it by k0
            if self.kgrid[0] < 0.0:
                self.kgrid[0] = min(self.k0)

        # Setup the grid of wave-vectors
        self.kvector, self._kbin_max = self._setup_grid_sphere(len(self.kgrid) * [self.dk],
                                                               self.kgrid, self.k0)
        
    @staticmethod
    def _setup_grid_sphere(dk, kgrid, k0):
        """
        Setup wave vector grid with spherical average (no symmetry),
        picking up vectors that fit into shells of width dk centered around
        the values specified in the input list kgrid.
        Returns a dictonary of lists of wavevectors, one entry for each element in the grid.
        """
        _log.info('setting up the wave-vector grid')
        kvec = defaultdict(list)
        # With elongated box, we choose the smallest k0 component to
        # setup the integer grid. This must be consistent with
        # expo_grid() otherwise it wont find the vectors
        kmax = kgrid[-1] + dk[-1]
        kbin_max = 1 + int(kmax / min(k0))
        kmax_sq = kmax**2

        # Choose iterator of spatial grid
        ndims = len(k0)
        if ndims == 3:
            _iterator = _sphere
        elif ndims == 2:
            _iterator = _disk
        else:
            raise ValueError('unsupported dimension {}'.format(ndims))
            
        for ik in _iterator(kbin_max):
            ksq = numpy.dot(k0*ik, k0*ik)
            if ksq > kmax_sq:
                continue
            # beware: numpy.sqrt is x5 slower than math one!
            knorm = math.sqrt(ksq)
            # Look for a shell of vectors in which the vector could fit.
            # This expression is general and allows arbitrary k grids
            # However, searching for the shell like this is not fast
            # (it costs about as much as the above)
            for ki, dki in zip(kgrid, dk):
                if abs(knorm - ki) < dki:
                    kvec[ki].append(tuple(ik + kbin_max))
                    break

        if len(kvec.keys()) != len(kgrid):
            _log.warning('some entries in the kgrid had no matching k-vector')

        return kvec, kbin_max

    def _decimate_k(self):
        """
        Pick up a random, unique set of nk vectors out ot the avilable
        ones without exceeding maximum number of vectors in shell
        nkmax.

        Return: the new list of k-norms and a list of random samples
        of k-vector indices.
        """
        # Setting the seed here once so as to get the same set
        # independent of filters.
        random.seed(1)
        kgrid = sorted(self.kvector.keys())
        selection = []
        for knorm in kgrid:
            nkmax = len(self.kvector[knorm])
            selection.append(random.sample(list(range(nkmax)), min(self.nk, nkmax)))
        return kgrid, selection

    @property
    def kvectors(self):
        """
        Dictionary of actual kvectors used to compute the correlation
        function

        The keys of the dictionary are the kgrid values, i.e. the
        average k values in each wave-vector shell. The values are
        lists of wavevectors.
        """
        db = defaultdict(list)
        for ik, knorm in enumerate(self.kgrid):
            for isel in self.selection[ik]:
                center_vec = numpy.array(self.kvector[knorm][isel]) - self._kbin_max
                db[knorm].append(list(self.k0 * center_vec))
        return db
    
    def report(self, verbose=False):
        """
        Return a formatted report of the wave-vector grid used to compute
        the correlation function

        The `verbose` option turns on writing of the individuals
        wavectors (also accessible via the `kvectors` property).
        """
        txt = '# k-point, average, std, vectors in shell\n'
        db = self.kvectors
        for knorm in db:
            knorms = []
            for kvec in db[knorm]:
                k_sq = numpy.dot(kvec, kvec)
                knorms.append(math.sqrt(k_sq))
            knorms = numpy.array(knorms)
            txt += "{} {:f} {:f} {}\n".format(knorm, knorms.mean(),
                                              knorms.std(),
                                              len(db[knorm]))
        if verbose:
            txt += '\n# k-point, k-vector\n'
            for knorm in db:
                for kvec in db[knorm]:
                    # Reformat numpy array
                    as_str = str(kvec)
                    as_str = as_str.replace(',', '')
                    as_str = as_str.replace('[', '')
                    as_str = as_str.replace(']', '')
                    txt += '{} {}\n'.format(knorm, as_str)
        return txt

    def _actual_k_grid(self):
        """
        Return exactly the average wave vectors from the selected ones

        Used to redefine the k grid.
        """
        k_grid = []
        for kk, knorm in enumerate(self.kgrid):
            av = 0.0
            for i in self.selection[kk]:
                av += _k_norm(self.kvector[knorm][i], self.k0, self._kbin_max)
            k_grid.append(av / len(self.selection[kk]))
        return k_grid
