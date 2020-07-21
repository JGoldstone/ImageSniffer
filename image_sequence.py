# -*- coding: utf-8 -*-
"""
Sequence of frames in a single directory
======================================

Class for identifying sequence(s) in a directory, frame ranges, etc.

"""

from pathlib import Path

__author__ = 'Joseph Goldstone'
__copyright__ = 'Copyright (C) 2020 Arnold & Richter Cine Technik GmbH & Co. Betriebs KG'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Joseph Goldstone'
__email__ = 'jgoldstone@arri.com'
__status__ = 'Experimental'


class ImageSequence:

    def __init__(self, dir_path, file_name, file_ext, start=0, end=1, inc=1, frame_digits=None):
        self._dir_path = dir_path
        self._file_name = file_name
        self._file_ext = file_ext
        # TODO check that start, end, inc are numeric
        self._start = start
        self._end = end # n.b. follows convention of 'range' &c in that it's one past the last element in a sequence
        self._inc = inc
        self._frame_digits = frame_digits

    def path_for_frame(self, frame):
        if not self._frame_digits:
            raise IndexError(f"path for frame {frame} requested, but image sequence's frame_digits attribute is None")
        p = Path(f"{self._dir_path}/{self._file_name}.{frame:0{self._frame_digits}}.{self._file_ext}")
        return p

    def sequences_in_dir(self):
        seqs = []
        f = self._start
        seq_start_frame = None
        last_seen = None
        looking_for_start = True
        while f < self._end:
            p = self.path_for_frame(f)
            if p.exists() and p.is_file():
                if looking_for_start:
                    seq_start_frame = f
                    looking_for_start = False
                last_seen = f
            else:
                if not looking_for_start:
                    seq_end_frame = last_seen
                    seqs.append((seq_start_frame, last_seen))
                    seq_start_frame = None
                    last_seen = None
                    looking_for_start = True
            f += self._inc
        if not looking_for_start and seq_start_frame is not None:
            seqs.append((seq_start_frame, last_seen))
        return seqs
