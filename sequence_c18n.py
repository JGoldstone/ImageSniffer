# -*- coding: utf-8 -*-
"""
Data structure to hold collections of simple statistics about tristimulus values fo the frames in a sequence
===================

Collect the data gathered by frame characterizations performed on each frame in a sequence.

"""
import datetime
import pandas as pd
from pathlib import Path
from fileseq import FileSequence

from frame_c18n import FrameC18n


def convert_mxf_clip_to_exr_frames(clip_path, seq_dir_path):
    print(f"someday this will convert the clip at {clip_path} to frames inside {seq_dir_path}")


class SequenceC18n(object):

    def __init__(self):
        self._volume = None
        self._classification = None
        self._owner = None
        self._show = None
        self._scene = None
        self._element_type = None
        self._reel = None
        self._sequences = []
        self._dataframe = None

    def parse_clip_path_or_seq_dir_path(self, path):
        parts = path.parts
        _, _, self._volume, self._classification, self._owner, self._show, self._scene, self._element_type,\
            self._reel, clip_file_or_seq_dir = parts
        if not self._volume.startswith('jgoldstone'):
            raise RuntimeError("sequence path volume must start with jgoldstone")
        if self._classification not in ['not_secret', 'arri_secret', 'other_secret']:
            raise RuntimeError("classification not one of 'not_secret', 'arri_secret', or 'other_secret'")
        if self._element_type not in ['ocd', 'derived', 'etc']:
            raise RuntimeError("element type must be one of 'ocd', 'derived', or 'etc'")
        if path.suffix in ['.mov', '.mxf']:
            raise RuntimeError("cannot currently directly characterize QuickTime or MXF clips")
        if path.suffix == '.ari':
            raise RuntimeError("cannot currently directly characterize ARRRIAW file")
        elif path.suffix == '.dpx':
            raise RuntimeError("cannot currently directly characterize ARRRIAW file")
        if not path.is_dir():
            raise RuntimeError("Only supported input for characterization is directory of ACES container files")
        sequences = FileSequence.findSequencesOnDisk(str(path))
        if not len(sequences):
            raise RuntimeError("The nominal directory of frames to characterize is empty")
        self._sequences = sequences

    def characterize_frames(self):
        columns = {}
        seq_start_time = datetime.datetime.now()
        for sequence in self._sequences:
            for i, _ in enumerate(sequence.frameSet()):
                frame = sequence[i]
                print(f"characterizing {frame}", end='...')
                frame_start_time = datetime.datetime.now()
                frame_c18n = FrameC18n(Path(frame))
                frame_c18n.tally()
                frame_end_time = datetime.datetime.now()
                print(frame_end_time - frame_start_time)
                frame_c18n.add_to_columns(columns)
        seq_end_time = datetime.datetime.now()
        print(f"total elapsed time for sequence c18n is {seq_end_time - seq_start_time}")
        self._dataframe = pd.DataFrame(columns)

    def save(self, path, empty_is_error=True, verbose=True):
        if not (len(self._dataframe.index) == 0 and len(self._dataframe.columns)) and empty_is_error:
            raise RuntimeError("sequence_c18n is empty")
        with open(path, 'w') as output_file:
            self._dataframe.to_csv(output_file)
            if verbose:
                print(f"saved {len(self._dataframe.index)} frame-c18ns ({len(self._dataframe.columns)} data per frame)")
