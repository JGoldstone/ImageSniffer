# -*- coding: utf-8 -*-
"""
Data structure to hold some simple statistics about tristimulus values
===================

Defines a class that provides data structures and methods for simple analysis
of the samples corresponding to, e.g., the pixels of an image. Such statistics include:
- multi-channel counters:
-- count of occurrences of all channel values having a zero value (i.e. pixel is black)
- single-channel counters:
-- component-wise count of negative-clip values
-- component-wise count of zero values
-- component-wise count of positive-clip values
- latches (registers of peak values):
-- component-wise largest-magnitude strictly negative values
-- component-wise smallest-magnitude strictly negative values for each channel
-- component-wise smallest-magnitude strictly positive values for each channel
-- component-wise largest-magnitude strictly positive values for each channel

"""

from collections import OrderedDict
import numpy as np

__author__ = 'Joseph Goldstone'
__copyright__ = 'Copyright (C) 2020 Arnold & Richter Cine Technik GmbH & Co. Betriebs KG'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Joseph Goldstone'
__email__ = 'jgoldstone@arri.com'
__status__ = 'Experimental'

__all__ = [
    'Registers'
]

def is_black_pixel(pixel):
    return pixel[0] == 0 & pixel[1] == 0 & pixel[2] == 0


def is_negative_clip_component(component):
    return component == np.finfo(np.dtype('f16')).min


def is_zero_component(component):
    return component == 0


def is_positive_clip_component(component):
    return component == np.finfo(np.dtype('f16')).max


def biggest_strictly_negative_non_clipping_value(array):
    return array[array < 0 & array > np.finfo(np.dtype('f16')).min].min()


def tiniest_strictly_negative_non_clipping_value(array):
    return array[array < 0 & array > np.finfo(np.dtype('f16')).min].max()


def tiniest_strictly_positive_non_clipping_value(array):
    return array[array > 0 & array < np.finfo(np.dtype('f16')).max].min()


def biggest_strictly_positive_non_clipping_value(array):
    return array[array > 0 & array < np.finfo(np.dtype('f16')).max].max()


class Counter(object):
    """Counter holding the number of pixels in an image for which a provided test function returned True
    """

    def __init__(self, desc, pred):
        """

        Parameters
        ----------
        desc : str
            Description of what is being counted
        pred : function
            Function taking a single argument, representing either a pixel or a pixel component
        """
        self.desc = desc
        self._pred = pred
        self.count = 0

    def tally_pixels(self, image, mask):
        self.count = len(np.argwhere(self._pred(image[mask])))

    def tally_channel_values(self, image, mask, channel):
        self.count = len(np.argwhere(self._pred(image[mask][..., channel])))

    def __str__(self):
        print(f"{self.desc}: {self.count}")


class Latch(object):
    """Register holding value representing the extrema of some function
    """

    def __init__(self, desc, func):
        """

        Parameters
        ----------
        desc : str
            description of value sought (e.g. 'tiniest observed strictly positive value')
        func : function
            function returning extreme of some channel value across an image
        """
        self.desc = desc
        self._func = func
        self.latched_value = None

    def latch_max_channel_value(self, image, mask, channel):
        self.latched_value = image[self._func(image, mask, channel)][channel]

    def __str__(self):
        print(f"{self.desc}: {self.latched_value}")


class Registers(object):
    """
    Groups sample-and-hold registers, where the groups are identified by name in the
    REGISTER_GROUP_ATTRIBUTES module-scope global variable, and there are as many
    members within each group as the image has channels.

    (n.b. not tested at the moment for anything but tristimulus colorimetry)

    Parameters
    ----------
    desc : str
        Descriptive string for register
    channel_names : array-like
        Names of image channels

    Attributes
    ----------
    _desc : str
        Descriptive string for register
    _channel_names : list
        list of unicode strings of names
    _pixel_counters : OrderedDict
        dictionary of counters of whole-pixel values (e.g. how many black pixels)
    _channel_counters : OrderedDict
        dictionary of counters of per-channel values (e.g. how many zero channel values)
    _latches : OrderedDict
        dictionary of latches of channel quantities (e.g. tiniest positive channel value)

    Methods
    -------
    _setup_pixel_counters
    _setup_channel_counters
    _setup_channel_latches
    tally
    """

    @staticmethod
    def _setup_pixel_counters(counters):
        for desc, func in [('black pixel count', is_black_pixel)]:
            counter = Counter(desc, func)
            counters[desc] = counter

    @staticmethod
    def _setup_channel_counters(channel_names, counters):
        for desc, func in [('negative clip', is_negative_clip_component),
                           ('zero', is_zero_component),
                           ('positive clip', is_positive_clip_component)]:
            for channel_name in channel_names:
                counter_name = f"{desc} ({channel_name})"
                counters[counter_name] = Counter(counter_name, func)

    @staticmethod
    def _setup_channel_latches(channel_names, latches):
        for desc, func in [('biggest strictly negative value', biggest_strictly_negative_non_clipping_value),
                           ('tiniest strictly negative value', tiniest_strictly_negative_non_clipping_value),
                           ('tiniest strictly positive value', tiniest_strictly_positive_non_clipping_value),
                           ('biggest strictly positive value', biggest_strictly_negative_non_clipping_value)]:
            for channel_name in channel_names:
                latch_name = f"{desc} ({channel_name})"
                latches[latch_name] = Latch(desc, func)

    def __init__(self, desc, channel_names):
        self._desc = desc
        self._channel_names = channel_names
        self._pixel_counters = OrderedDict()
        self._channel_counters = OrderedDict()
        self._latches = OrderedDict()
        self._setup_pixel_counters(self._pixel_counters)
        self._setup_channel_counters(self._channel_names, self._channel_counters)
        self._setup_channel_latches(self._channel_names, self._latches)

    def tally(self, img_array, ix_array):
        """
        Parameters
        ----------
        img_array : two-dimensional Numpy array of pixel values
            Holds the image whose pixels will be sent to the various registers for sampling
        ix_array : dictionary of two-dimensional numpy array of boolean values
            Matched in width and height to img_array, the values in the dictionary indicating whether
            the corresponding pixel passed some test.
        """
        for counter in self._pixel_counters.values():
            counter.tally_pixels(img_array, ix_array)
        for counter in self._channel_counters.values():
            for channel in self._channel_names:
                counter.tally_channel_values(img_array, ix_array, channel)
        for latch in self._latches.items.values():
            for channel in self._channel_names:
                latch.latch_max_channel_value(img_array, ix_array, channel)

    def __str__(self):
        print(f"{self._desc}:")
        for counter in self._pixel_counters:
            print(counter)
        for counter in self._channel_counters:
            print(counter)
        for latch in self._latches:
            print(latch)
