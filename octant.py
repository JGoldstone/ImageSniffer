# -*- coding: utf-8 -*-
"""
Data structure to characterize the distribution of pixel values within an octant.
===================

This class maintains a 3D census of tristimulus values within a volume representing
an octant in R(3), as well as a group of registers tracking basic statistics of the
values in the octant.


"""

import numpy as np

from registers import Registers
from bins import LogBins

# TODO generalize to support orthants

__author__ = 'Joseph Goldstone'
__copyright__ = 'Copyright (C) 2020 Arnold & Richter Cine Technik GmbH & Co. Betriebs KG'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Joseph Goldstone'
__email__ = 'jgoldstone@arri.com'
__status__ = 'Experimental'

__all__ = [
    'Octant'
]


class Octant:
    """

    """

    @staticmethod
    def octant_keys():
        keys = []
        for has_blue_neg in (True, False):
            for has_green_neg in (True, False):
                for has_red_neg in (True, False):
                    keys.append((has_red_neg, has_green_neg, has_blue_neg))
        return keys

    def __init__(self, img_spec, octant_key, most_neg, least_neg, num_bins):
        self._img_spec = img_spec
        self._octant_key = octant_key
        self.to_first_octant_scalars = np.diag([-1 if e else 1 for e in octant_key])
        self.cubelets = np.array((num_bins,) * 3, dtype=np.uint)
        self._registers = Registers(img_spec.nchannels)
        self._bins = [LogBins(most_neg, least_neg, num_bins)] * len(octant_key)

    def _pixels_in_octant(self, ixs):
        ix = np.full([self._img_spec.width, self._img_spec.height], True)
        for channel, channel_ix in enumerate(ixs.ix['neg']):
            ix = ix and channel_ix if self._octant_key[channel] else not channel_ix
        return ix

    def _update_bins_and_cubelets(self, img, ixs):
        img_ix_array = self._pixels_in_octant(ixs)
        for img_ix in np.argwhere(img_ix_array):
            first_octant_pixel = img[tuple(img_ix)] * self.to_first_octant_scalars
            cube_ix = []
            for i in range(self._img_spec.nchannels):
                bin_ix = self._bins[i].add_entry(first_octant_pixel[i])
                if bin_ix:  # do nothing if value was too tiny or too big
                    cube_ix.append(bin_ix)
            if len(cube_ix) == self._img_spec.nchannels:
                self.cubelets[tuple(cube_ix)] += 1

    def tally(self, img, ixs):
        self._update_bins_and_cubelets(img, ixs)
        self._registers.tally(img, ixs)
