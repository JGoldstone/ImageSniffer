# -*- coding: utf-8 -*-
"""
Repository for image data gathered from an individual frame
===================

Defines a class that collects information on the distribution of tristimulus
 samples in R3, both as a whole, and by octant.

"""
import OpenImageIO as oiio
from OpenImageIO import ImageInput

from registers import Registers
from octant import Octant


class FrameC18n:

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
        self._overall_registers = Registers(f"rewgisters for entire image", self._image_input.spec)
        self.octants = {}
        for octant in Octant.octant_keys():
            self.octants[octant] = Octant(self._image_input.spec(), octant, most_neg, least_neg, num_bins)

    def tally(self):
        img_array = self._image_input.read_image()
        ixs = Registers("Overall image statistics", self._image_input.spec().channel_names)
        print(f"starting overall image tally")
        self._overall_registers.tally(img_array, ixs)
        print(f"ending overall image tally")
        print(f"starting octant tallies")
        for octant in self.octants.values():
            print(f"starting tally for octant {octant}:")
            octant.tally(img_array)
        print(f"ending octant tallies")

    def __str__(self):
        desc = []
        img_size_desc = f"{self._img_size}-byte image, {self._width*self._height*2}"
        desc.append(img_size_desc)
        overall_bin_desc = str(self._overall_registers)
        desc.append(overall_bin_desc)
        for octant in self.octants:
            desc.append(str(octant))
        return '\n'.join(desc)
