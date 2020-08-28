# -*- coding: utf-8 -*-
"""
Repository for image data gathered from an individual frame
===================

Defines a class that collects information on the distribution of tristimulus
 samples in R3, both as a whole, and by octant.

"""
import numpy as np

import OpenImageIO as oiio
from OpenImageIO import ImageInput

from registers import Registers
from octant import Octant


class FrameC18n(object):

    def __init__(self, path, bin_min_exp=-4, bin_max_exp=2, num_bins=None):
        if not num_bins:
            num_bins = 1 + bin_max_exp - bin_min_exp
        self._path = path
        self._img_size = self._path.stat().st_size
        self._image_input = ImageInput.open(str(self._path))
        if self._image_input is None:
            # TODO check that the problem *is* actually the file is not there
            raise FileNotFoundError(f"could not read image `{self._path}': {oiio.geterror()}")
        roi = self._image_input.spec().roi
        # n.b. ROI xend and yend are range-style 'one beyond the end' values
        self._x = roi.xbegin
        self._width = roi.xend - roi.xbegin
        self._y = roi.ybegin
        self._height = roi.yend - roi.ybegin
        self._overall_registers = Registers(f"registers for entire image", "overall",
                                            self._image_input.spec().channelnames)
        self.octants = {}
        for octant_key in Octant.keys():
            self.octants[octant_key] = Octant(self._image_input.spec(), octant_key, bin_min_exp, bin_max_exp, num_bins)

    def tally(self):
        img_array = self._image_input.read_image()
        self._overall_registers.tally(img_array, np.full(img_array.shape[0:2], True))
        for octant in self.octants.values():
            octant.tally(img_array)

    def add_to_columns(self, columns):
        """Append the information in the frame c18n to a Pandas DataFrame

        Parameters
        ----------
        columns : dict

        """
        self._overall_registers.add_to_columns(columns)
        for octant in self.octants:
            self.octants[octant].add_to_columns(columns)

    def summarize(self, indent_level=0):
        summary = ''
        summary += "overall image statistics:\n"
        summary += self._overall_registers.summarize(indent_level + 1)
        for octant in self.octants.values():
            if samples_in_octant := octant.samples_in_octant:
                summary += f"{'  '*indent_level}statistics for {octant} ({samples_in_octant} samples):\n"
                summary += octant.summarize(indent_level + 1)
        return summary

    # def __str__(self):
    #     return f"Frame c18n of {Path(self._path).name} ({self._width}x{self._height})"
