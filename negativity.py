# -*- coding: utf-8 -*-
"""
RGB negative value characterization
=======================================

Examines OpenEXR images to look for negative pixel values, then characterize them

"""

import numpy as np

import OpenImageIO as oiio
from OpenImageIO import ImageInput, TypeUnknown
from math import log10


def lerp(x, min_domain, max_domain, min_range, max_range):
    return min_range + (max_range - min_range) * (x - min_domain) / (max_domain - min_domain)


class Bin:

    def __init__(self, min_binned, max_binned, num_bins):
        self._min_binned = min_binned
        self._max_binned = max_binned
        self._too_small_bin_count = 0
        self._too_large_bin_count = 0
        self._bins = np.array(num_bins, dtype=np.int)

    def add_entry(self, value):
        ix = lerp(log10(value), self._min_binned, self._max_binned, 0, len(self._bins)-1)
        if ix < 0:
            self._too_small_bin_count += 1
        elif ix > len(self._bins) - 1:
            self._too_large_bin_count += 1
        else:
            self._bins[ix] += 1


class ImageCharacterization:

    def __init__(self, path, num_bins=81):
        self._path = path
        self._perfect_black_count = 0
        self._octants = []
        self._octant_counts = []
        self._rgb_component_counts = np.zeros(3, dtype=np.int)
        self._total_outside_first_quadrant_count = 0
        self._min_binned = -4  # anything smaller in magnitude than 1.0e-4 is 'too small'
        self._max_binned = 1  # anything bigger in magnitude than 10 is 'too big'
        self._any_bin = Bin(self._min_binned, self._max_binned, num_bins)
        self._any_red_bin = Bin(self._min_binned, self._max_binned, num_bins)
        self._any_green_bin = Bin(self._min_binned, self._max_binned, num_bins)
        self._any_blue_bin = Bin(self._min_binned, self._max_binned, num_bins)

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

    def _add_to_any_bins(self, img, has_neg_red, has_neg_green, has_neg_blue):
        has_any_neg = has_neg_red | has_neg_green | has_neg_blue
        for rgb in img[has_any_neg]:
            self._any_bin.add_entry(rgb)
        for rgb in img[has_neg_red]:
            self._any_red_bin.add_entry(rgb)
        for rgb in img[has_neg_green]:
            self._any_green_bin.add_entry(rgb)
        for rgb in img[has_neg_blue]:
            self._any_blue_bin.add_entry(rgb)

    def _tally(self):
        image_input = ImageInput.open(self._path)
        if image_input is None:
            # TODO check that the problem *is* actually the file is not there
            raise FileNotFoundError(f"could not read image `{self._path}': {oiio.geterror()}")
        img = image_input.read_image(format=TypeUnknown)
        has_perfect_black = img[:, :, 0] == 0 & img[:, :, 1] == 0 & img[:, :, 2] == 0
        self._perfect_black_count = np.sum(has_perfect_black)
        has_neg_red = img[:, :, 0] < 0
        has_neg_green = img[:, :, 1] < 0
        has_neg_blue = img[:, :, 2] < 0
        self._add_to_any_counts(has_neg_red, has_neg_green, has_neg_blue)
        self._add_to_any_bins(img, has_neg_red, has_neg_green, has_neg_blue)
        self._add_to_octants(img, has_neg_red, has_neg_green, has_neg_blue)
