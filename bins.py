# -*- coding: utf-8 -*-
"""
Data structure for collecting a distribution of some variable into bins
===================

Defines a class that maps a continuous real-valued variable into a finite
set of bins representing disjoint intervals. The domains corresponding
to each bin are open at the 'low end' and closed at the 'high end'.
Domain values outside the given domain increment overflow or underflow
counters.

As an example application consider taking negative exposure values and
mapping them across bins representing really huge errors (-100 e.v.)
to tiny errors (barely more than 10e-6). Since we can't take the log of
a negative numberm we first negate the domain value and then take its
base 10 log, and use that to index a set of 8 bins (and beyond that,
overflow and underflow counters).

The bin would be created like this:
    # test_bin = LogBins(2, -4, 8)

Taking the base 10 log of the negated negative exposure value is done
at the moment by LogNBin's add_entry method.
# TODO generalize LogBin to Bin where taking the base 10 log of the negated EV is done by a ctor arg lambda

The distribution across bins (and into overflow and underflow counters)
of x, the base 10 log of a negated e.v., would be like this, given the
constructor call above.

    add to overflow counter if x > 2
    add to _bins[0] if 10e+1 < x <= 10e+2 [+2, +1)
    add to _bins[1] if 10e+0 < x <= 10e+1 [+1,  0)
    add to _bins[2] if 10e-1 < x <= 10e+0 [ 0, -1)
    add to _bins[3] if 10e-2 < x <= 10e-1 [-1, -2)
    add to _bins[4] if 10e-3 < x <= 10e-2 [-2, -3)
    add to _bins[5] if 10e-4 < x <= 10e-3 [-3, -4)
    add to _bins[6] if 10e-5 < x <= 10e-4 [-4, -5)
    add to _bins[7] if 10e-6 < x <= 10e-5 [-5, -6)
    add to underflow counter if x <= 10e-6

"""

from sys import float_info
from math import log10, floor

import numpy as np

__author__ = 'Joseph Goldstone'
__copyright__ = 'Copyright (C) 2020 Arnold & Richter Cine Technik GmbH & Co. Betriebs KG'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Joseph Goldstone'
__email__ = 'jgoldstone@arri.com'
__status__ = 'Experimental'

# TODO find out if class methods would show up in __all__
__all__ = [
    'LogBins'
]

def lerp(x, min_domain, max_domain, min_range, max_range):
    return min_range + (max_range - min_range) * (x - min_domain) / (max_domain - min_domain)


class LogBins:

    def __init__(self, log_max, log_min, num_bins=None):
        if num_bins is None:
            num_bins = log_max - log_min
        self.log_max = log_max
        self.log_min = log_min
        self.num_underflowed = 0
        self.num_overflowed = 0
        self._epsilon = float_info.epsilon * 4
        self._bins = np.zeros(num_bins, dtype=np.int)

    def ix_for_value(self, value):
        lerped = lerp(value, self.log_max, self.log_min, 0, len(self._bins))
        # handle case where lerped value is exactly self.max
        if lerped == len(self._bins):
            lerped -= 1
        return lerped

    def value_for_ix(self, ix):
        return lerp(ix, 0, len(self._bins), self.log_max, self.log_min)

    def add_entry(self, value):
        log_value = log10(value)
        if log_value > self.log_max:
            self.num_overflowed += 1
            return None
        elif log_value <= self.log_min:
            self.num_underflowed += 1
            return None
        else:
            ix = floor(self.ix_for_value(log_value))
            self._bins[ix] += 1
            return ix

    def bin_bounds(self, ix):
        assert 0 <= ix < len(self._bins)
        assert floor(ix) == ix
        lower_bound = 10 ** self.value_for_ix(ix)
        upper_bound = 10 ** self.value_for_ix(ix + 1)
        return lower_bound, upper_bound

    # TODO write a generator for bounds range strings that includes overflow and underflow counters in addition to bin bounds, using the infinity symbol wheere needed
