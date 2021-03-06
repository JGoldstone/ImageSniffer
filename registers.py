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
    """Finds black pixels in a 2D array of tristimulus values

    Parameters
    ----------
    pixel : numpy.ndarray

    Returns
    -------
    bool

    """
    return np.all(pixel == 0, axis=-1)


def is_negative_clip_component(component):
    return component == np.finfo(np.half).min


def is_zero_component(component):
    return component == 0


def is_positive_clip_component(component):
    return component == np.finfo(np.half).max


def strictly_negative_but_not_clipped_inverse_mask(array):
    return np.logical_and(array > np.finfo(np.float16).min, array < 0)


def biggest_strictly_negative_non_clipping_value(array):
    inverse_mask = strictly_negative_but_not_clipped_inverse_mask(array)
    if np.any(array[inverse_mask]):
        return array[inverse_mask].min()


def tiniest_strictly_negative_non_clipping_value(array):
    inverse_mask = strictly_negative_but_not_clipped_inverse_mask(array)
    if np.any(array[inverse_mask]):
        return array[inverse_mask].max()


def strictly_positive_but_not_clipped_inverse_mask(array):
    return np.logical_and(array > 0, array < np.finfo(np.float16).max)


def tiniest_strictly_positive_non_clipping_value(array):
    inverse_mask = strictly_positive_but_not_clipped_inverse_mask(array)
    if np.any(array[inverse_mask]):
        return array[inverse_mask].min()


def biggest_strictly_positive_non_clipping_value(array):
    inverse_mask = strictly_positive_but_not_clipped_inverse_mask(array)
    if np.any(array[inverse_mask]):
        return array[inverse_mask].max()


def add_to_columns(columns, keys_and_values):
    for key, value in keys_and_values:
        if key not in columns.keys():
            columns[key] = [value]
        else:
            columns[key] += [value]


class Counter(object):
    """Counter holding the number of pixels in an image for which a provided test function returned True
    """

    def __init__(self, desc, label, pred, channel=None):
        """

        Parameters
        ----------
        desc : str
            Description of what is being counted
        pred : function
            Function taking a single argument, representing either a pixel or a pixel component
        """
        self.desc = desc
        self._label = label
        self._pred = pred
        self._channel = channel
        self.count = None

    def tally_pixels(self, img, inv_mask):
        """Count number of times a per-pixel predicate is satisfied

        Parameters
        ----------
        img : np.ndarray
        inv_mask : np.ndarray
            2D array of booleans, where a True entry means 'tally this pixel'

        Returns
        -------

        """
        self.count = len(np.argwhere(self._pred(img[inv_mask])))

    def tally_channel_values(self, img, inv_mask):
        self.count = len(np.argwhere(self._pred(img[inv_mask][..., self._channel])))

    def add_to_columns(self, columns):
        add_to_columns(columns, [(self._label, self.count)])

    def summarize(self, indent_level=0):
        if self.count:
            return f"{'  '*indent_level}{self.desc}: {self.count}\n"

    def __str__(self):
        print(f"{self.desc}: {self.count}")


class Latch(object):
    """Register holding value representing the extrema of some function
    """

    def __init__(self, desc, label, func, channel):
        """

        Parameters
        ----------
        desc : str
            description of value sought (e.g. 'tiniest observed strictly positive value')
        func : function
            function returning extreme of some channel value across an image
        """
        self.desc = desc
        self._label = label
        self._func = func
        self._channel = channel
        self.values_examined_count = 0
        self.latched_value = 0

    def latch_max_channel_value(self, img, inv_mask):
        self.latched_value = self._func(img[inv_mask][..., self._channel])

    def summarize(self, indent_level=0):
        if self.latched_value:
            return f"{'  ' * indent_level}{self.desc}: {self.latched_value}\n"

    def add_to_columns(self, columns):
        latch_examined_label = f"{self._label}_examined"
        latch_value_label = self._label
        add_to_columns(columns, [(latch_examined_label, self.values_examined_count),
                                 (latch_value_label, self.latched_value)])

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

    def __init__(self, desc, domain, channel_names):
        self._desc = desc
        self._domain = domain
        self._channel_names = channel_names
        self._pixel_counters = OrderedDict()
        self._channel_counters = OrderedDict()
        self._latches = OrderedDict()
        self._setup_pixel_counters(domain, self._pixel_counters)
        self._setup_channel_counters(domain, self._channel_names, self._channel_counters)
        self._setup_channel_latches(domain, self._channel_names, self._latches)

    @staticmethod
    def _setup_pixel_counters(domain, counters):
        for desc, func_label, func in [('black pixel count', 'blk', is_black_pixel)]:
            counter_name = f"{desc}"
            counter_label = f"{domain}_{func_label}"
            counter = Counter(counter_name, counter_label, func)
            counters[func_label] = counter

    @staticmethod
    def _setup_channel_counters(domain, channel_names, counters):
        for desc, func_label, func in [('negative clip', 'nclp', is_negative_clip_component),
                           ('zero', 'zero', is_zero_component),
                           ('positive clip', 'pclp', is_positive_clip_component)]:
            for channel, channel_name in enumerate(channel_names):
                counter_name = f"{desc}_{channel_name}"
                counter_label = f"{domain}_{func_label}_{channel_name}"
                counters[counter_name] = Counter(counter_name, counter_label, func, channel)

    @staticmethod
    def _setup_channel_latches(domain, channel_names, latches):
        for desc, func_label, func in [('biggest strictly negative value', 'nbig', biggest_strictly_negative_non_clipping_value),
                                       ('tiniest strictly negative value', 'ntin', tiniest_strictly_negative_non_clipping_value),
                                       ('tiniest strictly positive value', 'ptin', tiniest_strictly_positive_non_clipping_value),
                                       ('biggest strictly positive value', 'pbig', biggest_strictly_positive_non_clipping_value)]:
            for channel, channel_name in enumerate(channel_names):
                latch_desc = f"{desc} {channel_name}"
                latch_label = f"{domain}_{func_label}_{channel_name}"
                latches[latch_desc] = Latch(latch_desc, latch_label, func, channel)

    def tally(self, img, inv_mask):
        """
        Parameters
        ----------
        img : two-dimensional Numpy array of pixel values
            Holds the image whose pixels will be sent to the various registers for sampling
        inv_mask : dictionary of two-dimensional numpy array of boolean values
            Matched in width and height to img, the values in the dictionary indicating whether
            the corresponding pixel passed some test.
        """
        for counter in self._pixel_counters.values():
            counter.tally_pixels(img, inv_mask)
        for counter in self._channel_counters.values():
            counter.tally_channel_values(img, inv_mask)
        for latch in self._latches.values():
            latch.values_examined_count = np.sum(inv_mask)
            latch.latch_max_channel_value(img, inv_mask)

    def add_to_columns(self, columns):
        for counter in self._pixel_counters.values():
            counter.add_to_columns(columns)
        for counter in self._channel_counters.values():
            counter.add_to_columns(columns)
        for latch in self._latches.values():
            latch.add_to_columns(columns)

    @staticmethod
    def _some_nonzero_counter_seen(counters):
        saw_nonzero = False
        for counter in counters:
            if counter.count:
                saw_nonzero = True
                break
        return saw_nonzero

    def summarize(self, indent_level=0):
        representation = ''
        if self._some_nonzero_counter_seen(self._pixel_counters.values()):
            representation += f"{'  '*indent_level}pixel counters:\n"
            for counter in self._pixel_counters.values():
                if counter_summary := counter.summarize(indent_level + 1):
                    representation += counter_summary
        if self._some_nonzero_counter_seen(self._channel_counters.values()):
            representation += f"{'  '*indent_level}channel counters:\n"
            for counter in self._channel_counters.values():
                if counter_summary := counter.summarize(indent_level + 1):
                    representation += counter_summary
        for latch in self._latches.values():
            if latch_summary := latch.summarize(indent_level + 1):
                representation += latch_summary
        return representation

    # def __str__(self):
    #     return self._desc
