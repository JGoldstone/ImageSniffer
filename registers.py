# -*- coding: utf-8 -*-
"""
Data structure to hold some simple statistics about tristimulus values
===================

Defines a class that provides data structures and methods for simple analysis
of the samples corresponding to, e.g., the pixels of an image. Such statistics include:
- component-wise maxima for each channel, where the maxima for a channel retains the
  context (that is, the values of all the other channels accompanying it when it was
  seen to be the maximum for that channel).
-- component-wise largest strictly positive values for each channel
-- component-wise smallest strictly positive values for each channel
-- component-wise smallest strictly negative values for each channel
-- component-wise largest strictly negative values for each channel
- component-wise count of occurrences of a channel having a zero value
- component-wise count of occurrences of a channel having the maximum possible value
  within the range a half-float can represent
- component-wise count of occurrences of a channel having the minimum possible value
  within the range a half-float can represent

"""

from collections import OrderedDict
import numpy as np
from sample_and_hold_register import SampleAndHoldRegister

__author__ = 'Joseph Goldstone'
__copyright__ = 'Copyright (C) 2020 Arnold & Richter Cine Technik GmbH & Co. Betriebs KG'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Joseph Goldstone'
__email__ = 'jgoldstone@arri.com'
__status__ = 'Experimental'

__all__ = [
    'Registers'
]


# Some predicates to use in constructing registers. "small" and "big" are in the sense of size, not position in R(1)
def _biggest_strictly_positive():
    return lambda candidate, reference: True if candidate > reference > 0 else False


def _smallest_strictly_positive():
    return lambda candidate, reference: True if reference > candidate > 0 else False


def _smallest_strictly_negative():
    return lambda candidate, reference: True if reference < candidate < 0 else False


def _biggest_strictly_negative():
    return lambda candidate, reference: True if candidate < reference < 0 else False


def _zero():
    return lambda candidate: candidate == 0


def _within_epsilon(target_value):
    return lambda candidate: abs(candidate - target_value) < 4 * np.finfo(np.float16).eps


def _always_true(_):
    return True


REGISTER_GROUP_ATTRIBUTES = OrderedDict([
    # group_name, ix_array_name, test, is_pure_counter
    ('negative clip count', ('neg_clip', _within_epsilon(np.finfo(np.float16).min), True)),
    ('biggest strictly negative value', ('neg', _biggest_strictly_negative(), False)),
    ('tiniest strictly negative value', ('neg', _smallest_strictly_negative(), False)),
    ('zero count', ('zero', _zero, True)),
    ('black count', ('black', _always_true, True)),
    ('tiniest strictly positive number', ('pos', _smallest_strictly_positive(), False)),
    ('biggest strictly positive number', ('pos', _biggest_strictly_positive(), False)),
    ('positive clip count', ('pos_clip', _within_epsilon(np.finfo(np.float16).max), True))
])


class Registers:
    """
    Groups sample-and-hold registers, where the groups are identified by name in the
    REGISTER_GROUP_ATTRIBUTES module-scope global variable, and there are as many
    members within each group as the image has channels.

    (n.b. not tested at the moment for anything but tristimulus colorimetry)

    Parameters
    ----------
    img_spec : OpenImageIO ImageSpec helper class instance

    Attributes
    ----------
    _channel_names : list of unicode
    _registers : dictionary
        Ordered map of register group names to registers. The keys to the map are
        a tuple of the name of the index array and the name of the channel within
        the register group.

    Methods
    -------
    _setup_registers
    tally
    """

    # preserve the order of the register groups while creating sets of registers, one set member for each channel
    @staticmethod
    def _setup_registers(img_spec, registers):
        for (group_name, ix_array_name, test, is_pure_counter) in REGISTER_GROUP_ATTRIBUTES:
            for (channel_num, channel_name) in enumerate(img_spec.channelNames):
                key = (group_name, channel_name)
                registers[key] = SampleAndHoldRegister(channel_num, img_spec.nchannels, test, is_pure_counter)

    def __init__(self, img_spec):
        self._channel_names = img_spec.channelNames
        self._registers = OrderedDict()
        self._setup_registers(img_spec, self._registers)

    def tally(self, img, ix_arrays):
        """
        Parameters
        ----------
        img : two-dimensional Numpy array of pixel values
            Holds the image whose pixels will be sent to the various registers for sampling
        ix_arrays : dictionary of two-dimensional numpy array of boolean values
            Matched in width and height to img, the values in the dictionary indicate whether
            the corresponding pixel passed some test. In something of a modularity failure,
            the keys of the ix_arrays are hard-coded here, but are also hard-coded in the
            ImageIndices class. Sorry, rookie mistake.
        """
        for key in self._registers.keys():
            (group_name, _) = key
            (ix_array_name, _, _) = REGISTER_GROUP_ATTRIBUTES[group_name]
            ix_array = ix_arrays[ix_array_name]
            pix_ix = np.argwhere(ix_array)
            register = self._registers[key]
            for (row, col) in pix_ix:
                # we dom't actually care, at this point, that OIIO's row 0 is the bottom orow of the frame, not the top
                # also not particularly trying to optimize memory patterns; it's effectively a sparse array (sorta).
                register.sample_pixel(img[row][col])
