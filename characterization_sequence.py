# -*- coding: utf-8 -*-
"""
Sequence of frames to be characterized
======================================

Class collecting image sequencing information, chacaterizations thereof, and ways to plot them

"""

from pathlib import Path
from image_characterization import ImageCharacterization
from image_sequence import ImageSequence

__author__ = 'Joseph Goldstone'
__copyright__ = 'Copyright (C) 2020 Arnold & Richter Cine Technik GmbH & Co. Betriebs KG'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Joseph Goldstone'
__email__ = 'jgoldstone@arri.com'
__status__ = 'Experimental'


class CharacterizationSequence:

    def __init__(self, dir_path, file_base, frame_number_width, first, last, missing_frames_ok=False):
        self._seq = ImageSequence(dir_path, file_base, "exr", first, last, 1, frame_number_width)
        self.frame_paths = []
        self.c18ns = []
        self.build_frame_list()

    def build_frame_list(self):
        dir_path = self._seq.dir
        if not dir_path.exists():
            raise FileNotFoundError(f"The sequence directory `{dir_path}' could not be found")
        if not dir_path.is_dir():
            raise NotADirectoryError(f"The file `{dir_path}' exists but is not a directory")
        for f in range(self._seq.start, self._seq.end, self._seq.inc):
            self.frame_paths.append(self._seq.path_for_frame(f))

    def build_c18n_list(self):
        for p in self.frame_paths:
            c18n = ImageCharacterization(str(p))
            self.c18ns.append(c18n)

    def take_octant_census(self):
        octant_counts_across_sequence = {}
        total_census = 0
        for c18n in self.c18ns:
            for octant in c18n.octant_counts.keys():
                if octant in octant_counts_across_sequence:
                    octant_counts_across_sequence[octant] += c18n.octant_counts[octant]
                else:
                    octant_counts_across_sequence[octant] = c18n.octant_counts[octant]
                total_census += c18n.octant_counts[octant]
        for octant in octant_counts_across_sequence.keys():
            print (f"{octant}: {octant_counts_across_sequence[octant]}\n")
        return self.c18ns
