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
from image_indices import ImageIndices

class FrameC18n:

    def __init__(self, path):
        self._path = path
        self._img_size = self._path.stat().st_size
        self._image_input = ImageInput.open(self._path)
        if self._image_input is None:
            # TODO check that the problem *is* actually the file is not there
            raise FileNotFoundError(f"could not read image `{self._path}': {oiio.geterror()}")
        spec = self._image_input.spec()
        roi = spec.roi
        # n.b. ROI xend and yend are range-style 'one beyond the end' values
        self._x = roi.xbegin
        self._width = roi.xend - roi.xbegin
        self._y = roi.ybegin
        self._height = roi.yend - roi.ybegin
        self._overall_registers = Registers()
        for octant in Octant.octant_keys():
            self.octants[octant] = Octant(octant)

    def _tally(self):
        img = self._image_input.read_image()
        ixs = ImageIndices(img)
        self._overall_registers.tally(img, ixs)
        for octant in self.octants:
            octant.tally(img, ixs)

    def __str__(self):
        desc = []
        img_size_desc = f"{self._img_size}-byte image, {self._width*self._height*2}"
        desc.append(img_size_desc)
        overall_bin_desc = str(self._overall_registers)
        desc.append(overall_bin_desc)
        for octant in self.octants:
            desc.append(str(octant))
        return '\n'.join(desc)
