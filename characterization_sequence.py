# -*- coding: utf-8 -*-
"""
Sequence of frames to be characterized
======================================

Class collecting image sequencing information, chacaterizations thereof, and ways to plot them

"""

from pathlib import Path
from image_characterization import ImageCharacterization

__author__ = 'Joseph Goldstone'
__copyright__ = 'Copyright (C) 2020 Arnold & Richter Cine Technik GmbH & Co. Betriebs KG'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Joseph GOldstone'
__email__ = 'jgoldstone@arri.com'
__status__ = 'Experimental'

class CharacterizationSequence:

    def __init__(self, dir_path, file_base, frame_number_width, first, last, missing_frames_ok=False):
        assert frame_number_width >= 0
        self._dir_path = dir_path
        self._file_base = file_base
        self._frame_number_width = frame_number_width
        self._suffix = 'exr'
        self._first = first
        self._last = last
        self._missing_frames_ok = missing_frames_ok
        self.frame_paths = []
        self.c18ns = []
        self.build_frame_list()

    def build_frame_list(self):
        dir_path = Path(self._dir_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"The file `{dir_path}' could not be found")
        if not dir_path.is_dir():
            raise NotADirectoryError(f"The file `{dir_path}' exists but is not a directory")
        frame_numbers = range(self._first, self._last + 1)
        for frame_number in frame_numbers:
            num_component = str(frame_number).rjust(self._frame_number_width, '0')
            if self._frame_number_width == 0:
                file_path = Path(f"{dir_path}/{self._file_base}.exr")
            else:
                file_path = Path(f"{dir_path}/{self._file_base}.{num_component}.exr")
            if not file_path.exists():
                raise FileNotFoundError(f"The file `{file_path}' could not be found")
            self.frame_paths.append(file_path)

    def characterize_frames(self):
        paths = [str(fp) for fp in self.frame_paths]
        for path in paths:
            self.c18ns.append(ImageCharacterization(path))
            print(f"characterized `{path}'")
