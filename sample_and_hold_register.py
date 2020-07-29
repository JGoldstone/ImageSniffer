# -*- coding: utf-8 -*-
"""
Sample-and-hold register
===================

Defines a register that can be fed samples, which are tested against some predicate
supplied at the time the register is constructed. If the test returns True, then
the sampled value is retained.

"""

__author__ = 'Joseph Goldstone'
__copyright__ = 'Copyright (C) 2020 Arnold & Richter Cine Technik GmbH & Co. Betriebs KG'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Joseph Goldstone'
__email__ = 'jgoldstone@arri.com'
__status__ = 'Experimental'

__all__ = [
    SampleAndHold Register, sample_pixel
]

class SampleAndHoldRegister:
    """
    Defines a register that, repeatedly fed pixels and an index into that pixel's components (a 'channel'),
    holds the 'peak' observed pixel channel value presented to it, along with the entire pixel as the
    context in which the pixel was observed.

    Parameters
    ----------
    channel : int
        Index into the components of the pixel, indicating which component is being sampled
    test : function of two required arguments
        Function taking two values and returning a boolean.
    desc : unicode
        Description of the quantity the register holds, e.g. 'tiniest strictly positive value'
    count_only : boolean
        Indicates there's no need to maintain a reference sample; it suffices to count the
        number of times the sample_pixel method has been called.

    Attributes
    ----------
    _channel : int
        Index into the pixel that accesses the desired pixel component
    _test : function of two required arguments
        Function that is cqlled with the pixel component indicated by the channel argument
        as its first argument and the last previous pixel component value that satisfied this
        predicate as the second argument. It is used to determine whether the cadidate new value
        should replae teh stored reference value.
    _desc : unicode
        Description of the quantity the register holds
    _count-only : boolean
        Indicator that only a count of the number of samples is required; neither testing nor
        storing of context should occur
    _sample_count : int
        Number of times this register has been fed samples
    _channel_value : numeric
        Initially None, updated with pixel channel values that 'pass the test'
    _context_pixel : list of numeric
        List of pixel values for the pixel whose examined channel value 'passed the test'

    Methods
    -------
    __str__
    sample_pixel

    """
    # channel needs to be int between 0 and num_channels - 1
    def __init__(self, channel, test, desc, count_only=False):
        self._channel = channel
        self._test = test
        self._desc = desc
        self._count_only = count_only
        self._sample_count = 0
        self._channel_value = None
        self._context_pixel = None

    def sample_pixel(self, pixel):
        candidate_channel_value = pixel[self._channel]
        if not self._count_only:
            if not self._context_pixel or self._test(candidate_channel_value, self._channel_value):
                self._channel_value = candidate_channel_value
                self._context_pixel = pixel
        self._sample_count += 1

    def __str__(self):
        if self._count_only:
            return f"{self._desc}: {self._sample_count} occurrences seen"
        else:
            return f"{self._desc}: {self._channel_value} in channel {self._channel} of value [{self._context_pixel}]"
