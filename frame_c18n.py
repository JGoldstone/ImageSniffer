# -*- coding: utf-8 -*-
"""
Repository for image data gathered from an individual frame
===================

Defines a class that collects information on the distribution of tristimulus
 samples in R3, both as a whole, and by octant.

"""
import numpy as np
from pathlib import Path

import OpenImageIO as oiio
from OpenImageIO import ImageInput

from registers import Registers
from octant import Octant


class FrameC18n(object):

    def __init__(self, path, most_neg=2, least_neg=-4, num_bins=None):
        if not num_bins:
            num_bins = 1 + most_neg - least_neg
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
        self._overall_registers = Registers(f"registers for entire image", self._image_input.spec().channelnames)
        self.octants = {}
        for octant in Octant.octant_keys():
            self.octants[octant] = Octant(self._image_input.spec(), octant, most_neg, least_neg, num_bins)

    def tally(self):
        img_array = self._image_input.read_image()
        self._overall_registers.tally(img_array, np.full(img_array.shape[0:2], True))
        for octant in self.octants.values():
            octant.tally(img_array)

    def summarize(self, indent_level=0):
        summary = ''
        summary += "overall image statistics:\n"
        summary += self._overall_registers.summarize(indent_level + 1)
        for octant in self.octants.values():
            if octant.samples_in_octant:
                summary += f"{'  '*indent_level}statistics for {octant}:\n"
                summary += octant.summarize(indent_level + 1)
        return summary

    def __str__(self):
        return f"Frame c18n of {Path(self._path).name} ({self._width}x{self._height})"
