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


class Octant(object):
    """
    Class to hold spatial and statistical distribution of pixel values in a Cartesian octant.

    Parameters
    ----------
    img_spec : OpenImageIO ImageSpec
        Describes the image to be analyzed
    octant_key : tuple of booleans
        Indicates whether an octant axis represents, in the space of pixel values, distance
        along the negative extent of that axis. (All octant data is stored with spatial
        and statistical data transformed to all-positive values; for the spatial data, the
        _to_first_octant_scalars attribute provides for transformatipon back to the original
        coordinate system.
    bin_min_exp : int
        Base-10 exponent of the largest negative value considered to not be 'overflow'. See
        documentation of the LogBins class for the gory details.
    bin_max_exp : int
        Base-10 exponent of the tinyest negative value considered to not be 'underflow'. See
        documentaionm of the LogBinds class for the gory details.
    num_bins : int
        Number of bins into which pixel values will be mapped
    """

    @staticmethod
    def keys():
        keys = []
        for has_blue_neg in (False, True):
            for has_green_neg in (False, True):
                for has_red_neg in (False, True):
                    keys.append((has_red_neg, has_green_neg, has_blue_neg))
        return keys

    def _name(self, channel_names, spacer):
        name_pieces = []
        negativities_and_names = zip(self._octant_key, channel_names)
        for negativity, name in negativities_and_names:
            name_pieces += [f"{name}{'-' if negativity else '+'}"]
        return spacer.join(name_pieces)

    def name(self):
        return self._name(['red', 'green', 'blue'], ', ')

    def label(self):
        return f"oct_{self._name(['r', 'g', 'b'], '')}"

    def _ix_for_octant(self, img_array):
        ix = np.full(img_array.shape[:2], True, dtype=np.bool)
        for chan, axis_negative_in_octant in enumerate(self._octant_key):
            chan_in_octant_ix = img_array[..., chan] < 0 if axis_negative_in_octant else img_array[..., chan] >= 0
            np.logical_and(ix, chan_in_octant_ix, ix)
        return ix

    @staticmethod
    def _log10_edges(min_exp, max_exp, num_bins=None):
        if not num_bins:
            num_bins = 1 + max_exp - min_exp
        zero_anchor = np.array([0])
        edge = 10**(np.linspace(min_exp, max_exp, num_bins, dtype=np.dtype('half')))
        max_anchor = np.array([np.finfo('half').max])
        edge = np.hstack([zero_anchor, edge, max_anchor])
        return edge

    def __init__(self, img_spec, octant_key, bin_min_exp, bin_max_exp, num_bins):
        self._img_spec = img_spec
        self._octant_key = octant_key
        self._to_first_octant_scalars = [-1 if e else 1 for e in octant_key]
        self._edge = self._log10_edges(bin_min_exp, bin_max_exp, num_bins=num_bins)
        self.samples_in_octant = 0
        self.hist3d = np.zeros((num_bins, num_bins, num_bins), dtype=np.uint)
        self._label = self.label()
        register_desc = f"registers for octant {self._label}"
        self._registers = Registers(register_desc, self._label, self._img_spec.channelnames)

    def _bin(self, img_array, octant_ix):
        bins = [self._edge] * 3
        img_in_octant = img_array[octant_ix]
        img_in_octant *= self._to_first_octant_scalars
        self.hist3d, _ = np.histogramdd(img_in_octant, bins)

    def tally(self, img_array):
        octant_ix = self._ix_for_octant(img_array)
        self.samples_in_octant = np.sum(octant_ix)
        self._bin(img_array, octant_ix)
        self._registers.tally(img_array, octant_ix)

    def add_to_columns(self, columns):
        self._registers.add_to_columns(columns)

    def summarize(self, indent_level=0):
        return self._registers.summarize(indent_level)

    # def __str__(self):
    #     representation = f"octant {self.name()}"
    #     return representation
