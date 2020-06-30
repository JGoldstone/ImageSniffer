# -*- coding: utf-8 -*-
"""
RGB negative value characterization
===================================

Examines OpenEXR images to look for negative pixel values, then characterizes them

-   :class:'image_characteristics.Charancterization

"""

import numpy as np

import OpenImageIO as oiio
from OpenImageIO import ImageInput
from sys import float_info
from math import log10, floor

__author__ = 'Joseph Goldstone'
__copyright__ = 'Copyright (C) 2020 Arnold & Richter Cine Technik GmbH & Co. Betriebs KG'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Joseph GOldstone'
__email__ = 'jgoldstone@arri.com'
__status__ = 'Experimental'


def lerp(x, min_domain, max_domain, min_range, max_range):
    return min_range + (max_range - min_range) * (x - min_domain) / (max_domain - min_domain)


class LogBin:

    def __init__(self, min_value, max_value, num_bins):
        self.min = min_value
        self.max = max_value
        self.num_underflowed = 0
        self.num_overflowed = 0
        self._epsilon = float_info.epsilon * 4
        self._bins = np.array(num_bins, dtype=np.int)

    def _fwd_lerp_value_to_ix(self, value):
        return floor(lerp(value, self.min, self.max - self._epsilon, 0, len(self._bins) - 1))

    def _inv_lerp_ix_to_value(self, ix):
        return lerp(ix, 0, len(self._bins) - 1, self.min, self.max - self._epsilon)

    def add_entry(self, value):
        assert value > 0
        ix = self._fwd_lerp_value_to_ix(log10(value))
        if ix < 0:
            self.num_underflowed += 1
        elif ix > len(self._bins) - 1:
            self.num_overflowed += 1
        else:
            self._bins[ix] += 1

    def bin_bounds(self, ix):
        assert 0 <= ix < len(self._bins)
        assert floor(ix) == ix
        lower_bound = 10 ** self._inv_lerp_ix_to_value(ix)
        upper_bound = 10 ** self._inv_lerp_ix_to_value(ix + 1)
        return lower_bound, upper_bound


class ImageCharacterization:

    def __init__(self, path, num_bins=81):
        self._path = path
        # We adopt the OpenImageIO convention of the origin being in the upper left
        self._left = 0
        self._top = 0
        self._width = 0
        self._height = 0
        self._perfect_black_count = 0
        self._octants = []
        self._octant_counts = []
        self._rgb_component_counts = np.zeros(3, dtype=np.int)
        self._total_outside_first_quadrant_count = 0
        self._min_binned = -4  # anything smaller in magnitude than 1.0e-4 is 'too small'
        self._max_binned = 1  # anything bigger in magnitude than 10 is 'too big'
        self._any_bin = LogBin(self._min_binned, self._max_binned, num_bins)
        self._any_red_bin = LogBin(self._min_binned, self._max_binned, num_bins)
        self._any_green_bin = LogBin(self._min_binned, self._max_binned, num_bins)
        self._any_blue_bin = LogBin(self._min_binned, self._max_binned, num_bins)
        self._tally()

    @staticmethod
    def _octant_for_rgb(rgb):
        octant = 0
        octant += 4 if rgb[0] < 0 else 0
        octant += 2 if rgb[1] < 0 else 0
        octant += 1 if rgb[2] < 0 else 0
        return octant

    def _add_to_any_counts(self, has_neg_red, has_neg_green, has_neg_blue):
        has_any_neg = has_neg_red | has_neg_green | has_neg_blue
        self._total_outside_first_quadrant_count = np.sum(has_any_neg)

    def _add_to_octants(self, img, has_neg_red, has_neg_green, has_neg_blue):
        has_only_red_neg = has_neg_red & ~has_neg_blue & ~has_neg_green
        has_only_green_neg = ~has_neg_red & has_neg_green & ~has_neg_blue
        has_only_blue_neg = ~has_neg_red & ~has_neg_green & has_neg_blue
        has_only_red_and_green_neg = has_neg_red & has_neg_green & ~has_neg_blue
        has_only_red_and_blue_neg = has_neg_red & ~has_neg_green & has_neg_blue
        has_only_green_and_blue_neg = ~has_neg_red & has_neg_green & has_neg_blue
        has_all_neg = has_neg_red & has_neg_green & has_neg_blue
        self._octants.append(img[has_only_blue_neg])
        self._octants.append(img[has_only_green_neg])
        self._octants.append(img[has_only_green_and_blue_neg])
        self._octants.append(img[has_only_red_neg])
        self._octants.append(img[has_only_red_and_blue_neg])
        self._octants.append(img[has_only_red_and_green_neg])
        self._octants.append(img[has_all_neg])
        for octant in self._octants:
            octant_census = len(octant)
            self._octant_counts.append(octant_census)
            if octant_census > 0:
                examplar = octant[0]
                for i in range(3):
                    if examplar[i] < 0:
                        self._rgb_component_counts[i] += octant_census

    def _add_to_any_log_bins(self, img, has_neg_red, has_neg_green, has_neg_blue):
        has_any_neg = has_neg_red | has_neg_green | has_neg_blue
        for rgb in img[has_any_neg]:
            self._any_bin.add_entry(rgb)
        for rgb in img[has_neg_red]:
            self._any_red_bin.add_entry(rgb)
        for rgb in img[has_neg_green]:
            self._any_green_bin.add_entry(rgb)
        for rgb in img[has_neg_blue]:
            self._any_blue_bin.add_entry(rgb)

    def _nuke_coords_from_ndarray_coords(self, x, y):
        return x, (self._height - 1) - y

    # add characterizations of probably clipped pixels at top and bottom

    def _tally(self):
        image_input = ImageInput.open(self._path)
        if image_input is None:
            # TODO check that the problem *is* actually the file is not there
            raise FileNotFoundError(f"could not read image `{self._path}': {oiio.geterror()}")
        img = image_input.read_image()
        has_zero_red = img[:, :, 0] == 0
        has_zero_green = img[:, :, 1] == 0
        has_zero_blue = img[:, :, 2] == 0
        has_perfect_black = has_zero_red & has_zero_green & has_zero_blue
        self._perfect_black_count = np.sum(has_perfect_black)
        has_neg_red = img[:, :, 0] < 0
        has_neg_green = img[:, :, 1] < 0
        has_neg_blue = img[:, :, 2] < 0
        neg_red_count = np.sum(has_neg_red)
        neg_green_count = np.sum(has_neg_green)
        neg_blue_count = np.sum(has_neg_blue)
        if neg_red_count > 0 and neg_green_count > 0 and neg_blue_count > 0:
            return
        self._add_to_any_counts(has_neg_red, has_neg_green, has_neg_blue)
        self._add_to_any_log_bins(img, has_neg_red, has_neg_green, has_neg_blue)
        self._add_to_octants(img, has_neg_red, has_neg_green, has_neg_blue)
