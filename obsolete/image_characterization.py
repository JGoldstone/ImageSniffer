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
__maintainer__ = 'Joseph Goldstone'
__email__ = 'jgoldstone@arri.com'
__status__ = 'Experimental'


class ImageCharacterization:

    def __init__(self, path, num_bins=81):
        self._path = path
        # We adopt the OpenImageIO convention of the origin being in the upper left
        self._left = 0
        self._top = 0
        self._width = 0
        self._height = 0
        self.perfect_black_count = 0
        self._octants = {}
        self.octant_counts = {}
        self._rgb_component_counts = np.zeros(3, dtype=np.int)
        self.total_outside_first_octant = 0
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

    # def _add_to_any_counts(self, has_neg_red, has_neg_green, has_neg_blue):
    #     has_any_neg = has_neg_red | has_neg_green | has_neg_blue
    #     self.total_outside_first_octant = np.sum(has_any_neg)

    def _tally_by_octant(self):
        for has_blue_neg in (True, False):
            for has_green_neg in (True, False):
                for has_red_neg in (True, False):
                    if not (has_red_neg or has_green_neg or has_blue_neg):
                        continue
                    key = (has_red_neg, has_green_neg, has_blue_neg)
                    if self._octants[key].any():
                        count = len(self._octants[key])
                        self.octant_counts[key] = count
                        self.total_outside_first_octant += count
                    else:
                        self.octant_counts[key] = 0

    def _add_to_octants(self, img, has_neg_red, has_neg_green, has_neg_blue):
        has_only_red_neg = has_neg_red & ~has_neg_blue & ~has_neg_green
        has_only_green_neg = ~has_neg_red & has_neg_green & ~has_neg_blue
        has_only_blue_neg = ~has_neg_red & ~has_neg_green & has_neg_blue
        has_only_red_and_green_neg = has_neg_red & has_neg_green & ~has_neg_blue
        has_only_red_and_blue_neg = has_neg_red & ~has_neg_green & has_neg_blue
        has_only_green_and_blue_neg = ~has_neg_red & has_neg_green & has_neg_blue
        has_all_neg = has_neg_red & has_neg_green & has_neg_blue
        self._octants[(False, False, True)] = img[has_only_blue_neg]
        self._octants[(False, True, False)] = img[has_only_green_neg]
        self._octants[(False, True, True)] = img[has_only_green_and_blue_neg]
        self._octants[(True, False, False)] = img[has_only_red_neg]
        self._octants[(True, False, True)] = img[has_only_red_and_blue_neg]
        self._octants[(True, True, False)] = img[has_only_red_and_green_neg]
        self._octants[(True, True, True)] = img[has_all_neg]
        self._tally_by_octant()

    def _octant_census_entry(self, title, key):
        count = self.octant_counts[key]
        total = self.total_outside_first_octant
        return f"{title}: {count} pixels ({100*count/total:4.1f}%)"

    def _octant_census(self):
        result = f"\tSingly negative:\n"
        yellow = self._octant_census_entry("\t\tsuper-yellow (only blue negative)", (False, False, True)) + '\n'
        magenta = self._octant_census_entry("\t\tsuper-magenta (only green negative)", (False, True, False)) + '\n'
        cyan = self._octant_census_entry("\t\tsuper-cyan (only red negative)", (True, False, False)) + '\n'
        result += (yellow + magenta + cyan)
        result += f"\tDoubly negative:\n"
        red = self._octant_census_entry("\t\tsuper-red (only red positive)", (False, True, True)) + '\n'
        green = self._octant_census_entry("\t\tsuper-green (only green positive)", (True, False, True)) + '\n'
        blue = self._octant_census_entry("\t\tsuper-blue (only blue positive)", (True, True, False)) + '\n'
        result += (red + green + blue)
        result += f"\tAll components negative:\n"
        black = self._octant_census_entry("\t\tsuper-black (all of red, green and blue negative", (True, True, True)) + '\n'
        result += black
        return result

    def _add_to_any_log_bins(self, img, has_neg_red, has_neg_green, has_neg_blue):
        has_any_neg = has_neg_red | has_neg_green | has_neg_blue
        # for rgb in img_array[has_any_neg]:
        #     self._any_bin.add_entry(rgb)
        for rgb in img[has_neg_red]:
            self._any_red_bin.add_entry(-rgb[0])
        for rgb in img[has_neg_green]:
            self._any_green_bin.add_entry(-rgb[1])
        for rgb in img[has_neg_blue]:
            self._any_blue_bin.add_entry(-rgb[2])

    def _nuke_coords_from_ndarray_coords(self, y, x):
        return x, (self._height - 1) - y

    # add characterizations of probably clipped pixels at top and bottom

    def _print_negative_pixel_components(self, img, has_neg_component, channel_name):
        for row in range(self._height):
            for col in range(self._width):
                if has_neg_component[row][col]:
                    nuke_x, nuke_y = self._nuke_coords_from_ndarray_coords(row, col)
                    print(f"{channel_name} negative at [{row}][{col}] (nuke({nuke_x}, {nuke_y})): {img[row,col,0]}, {img[row,col,1]}, {img[row,col,2]}")

    def _tally(self):
        image_input = ImageInput.open(self._path)
        if image_input is None:
            # TODO check that the problem *is* actually the file is not there
            raise FileNotFoundError(f"could not read image `{self._path}': {oiio.geterror()}")
        spec = image_input.spec()
        roi = spec.roi
        # n.b. ROI xend and yend are range-style 'one beyond the end' values
        self._x = roi.xbegin
        self._width = roi.xend - roi.xbegin
        self._y = roi.ybegin
        self._height = roi.yend - roi.ybegin
        img = image_input.read_image()
        has_zero_red = img[:, :, 0] == 0
        has_zero_green = img[:, :, 1] == 0
        has_zero_blue = img[:, :, 2] == 0
        has_perfect_black = has_zero_red & has_zero_green & has_zero_blue
        self.perfect_black_count = np.sum(has_perfect_black)
        has_neg_red = img[:, :, 0] < 0
        has_neg_green = img[:, :, 1] < 0
        has_neg_blue = img[:, :, 2] < 0
        neg_red_count = np.sum(has_neg_red)
        # if neg_red_count > 0:
        #     self._print_negative_pixel_components(img_array, has_neg_red, "red")
        neg_green_count = np.sum(has_neg_green)
        # if neg_green_count > 0:
        #     self._print_negative_pixel_components(img_array, has_neg_green, "green")
        neg_blue_count = np.sum(has_neg_blue)
        # if neg_blue_count > 0:
        #     self._print_negative_pixel_components(img_array, has_neg_blue, "blue")
        if neg_red_count > 0 or neg_green_count > 0 or neg_blue_count > 0:
            # self._add_to_any_counts(has_neg_red, has_neg_green, has_neg_blue)
            self._add_to_any_log_bins(img, has_neg_red, has_neg_green, has_neg_blue)
            self._add_to_octants(img, has_neg_red, has_neg_green, has_neg_blue)

    def __str__(self):
        """
        Returns a formatted string representation of the characterization

        Returns
        -------
        unicode
            Formatted characterization representation
        """
        black_count = f"Perfect black count: {self.perfect_black_count}\n"
        octant_census = self._octant_census() + '\n'
        return black_count + octant_census
