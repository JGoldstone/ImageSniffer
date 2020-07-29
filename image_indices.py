# -*- coding: utf-8 -*-
"""
Data structure to hold numpy arrays of boolean indices.
===================

Defines a class that holds numpy arrays resulting from applying boolean predicates
to an image. When used by the ImageSniffer, without such caching, these arrays would
need to be constructed nine times, once for an overall sampling, and once each for
eight per-octant sampling. (Even more if we support orthants someday.)

"""
__author__ = 'Joseph Goldstone'
__copyright__ = 'Copyright (C) 2020 Arnold & Richter Cine Technik GmbH & Co. Betriebs KG'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Joseph Goldstone'
__email__ = 'jgoldstone@arri.com'
__status__ = 'Experimental'

import numpy as np


class ImageIndices:

    def __init__(self, img):
        self.ix = {}
        self.ix['neg_clip'] = img == np.finfo(np.dtype('f16')).min
        self.ix['neg'] = img < 0
        self.ix['zero'] = img == 0
        self.ix['pos'] = img > 0
        self.ix['pos_clip'] = img == np.finfo(np.dtype('f16')).max
        self.ix['black'] = np.all(img == 0, axis=1)
