# -*- coding: utf-8 -*-
"""
Sequence of frames in a single directory
======================================

Class for identifying sequence(s) in a directory, frame ranges, etc.

"""
import re
from pathlib import Path
from itertools import groupby
from operator import itemgetter
from fileseq import findSequencesOnDisk

__author__ = 'Joseph Goldstone'
__copyright__ = 'Copyright (C) 2020 Arnold & Richter Cine Technik GmbH & Co. Betriebs KG'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Joseph Goldstone'
__email__ = 'jgoldstone@arri.com'
__status__ = 'Experimental'

SEQ_SUFFIXES = ['ari', 'dpx']
CLIP_SUFFIXES = ['mov', 'mxf']
SUFFIXES = SEQ_SUFFIXES.append(CLIP_SUFFIXES)


def _contiguous_ranges(frame_numbers):
    """Yields range objects the union of which bijectively maps to a set of supplied integers

    Parameters
    ----------
    frame_numbers : iterable
        sequence of integer values

    Returns
    ------
    List
        List of range objects each covering a contiguous, non-overlapping set of frames, with no adjacent ranges

    References
    ----------
        "Detecting consecutive integers in a list",
        https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
        TODO shorten _contiguous_ranges by only accessing first element and size of map (if possible)

    """
    ranges = []
    # TODO (hard) modify _contiguous_frame_number_range_generator to detect non-unit frame increment(s)
    for k, g in groupby(enumerate(frame_numbers), lambda ix: ix[0] - ix[1]):
        elems = []
        for thing in g:
            elem = thing[1]
            elems.append(elem)
        ranges.append(range(elems[0], elems[0] + len(elems)))
    return ranges






class ImageSequence:

    def __init__(self, dir_path, name, suffix, frame_range, frame_digits):
        """Create ImageSequence; values of None for name, suffix, start, end or frameD_digits mean the field is wildcard
        """
        self.dir_path = Path(dir_path)
        self.name = name
        self.suffix = suffix
        self.frame_range = frame_range
        self.frame_digits = frame_digits

    def path_for_frame(self, frame):
        if not self.frame_digits:
            raise IndexError(f"path for frame {frame} requested, but image sequence's frame_digits attribute is None")
        p = self.dir_path / f"{self.name}.{frame:0{self.frame_digits}}.{self.suffix}"
        return p

    @staticmethod
    def family_member_from_components(components):
        integer_pattern = r" *([0-9]+)"  # urk, carelessness w/leading spaces by writer allowed
        integer_regex = re.compile(integer_pattern)
        clan_name = None
        frame_width = None
        frame_number = None
        if len(components) == 1:
            frame_number_match = integer_regex.match(components[0])
            if frame_number_match:
                # 14.dpx
                frame_width = len(frame_number_match.group())
            else:
                # very common case: my_birthday_party.mxf, for example
                # also very common: my_frame_grab.dpx
                clan_name = components[0]
        else:
            frame_number_match = integer_regex.match(components[-1])
            if frame_number_match:
                #   foo.03.exr
                #   will.i.am.3.mxf
                #   sometimes_bad_things_happen. 17.dpx
                clan_name = components[0]
                frame_width = len(frame_number_match.group())
                frame_number = int(frame_number_match.group())
            else:
                # don't get involved with these
                #   mr.smith.dpx
                #   waiting.4.godot.mov
                clan_name = '.'.join(components)
        return clan_name, frame_width, frame_number

    @staticmethod
    def image_sequences_in_dir(dir_path):
        sequences = []
        for suffix in SEQ_SUFFIXES:
            # first pass: categorize all paths of a given suffix by base name
            # n.b. this excludes files or directories of the form '.foo', i.e.
            # classic 'hidden' files or directories
            clans_with_suffix = {}
            paths = Path(dir_path).glob(f"*+(.*?).{suffix}")
            for path in paths:
                (clan_name, frame_width, frame_number) \
                    = ImageSequence.family_member_from_components(path.stem.split('.'))
                if clan_name not in clans_with_suffix:
                    clans_with_suffix[clan_name] = {}
                clans_with_suffix[clan_name][frame_width].append(frame_number)
            for (clan_name, families) in clans_with_suffix.items():
                for (frame_width, frame_numbers) in families.items():
                    for frame_range in _contiguous_ranges(frame_numbers):
                        sequences.append(ImageSequence(dir_path, clan_name, suffix, frame_range,
                                                       frame_digits=frame_width))
        return sequences

