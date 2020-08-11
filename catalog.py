# -*- coding: utf-8 -*-
"""
Catalog of image locations, characteristics and statisical information
===================

Defines a class that can build and maintain a catalog of image information,
including the names and location of image sequences (including single-image
sequences), their characteristics (e.g. essence colorspace, frame range),
representative images, and statistical information (negative pixel c18ns,
FALL, MaxFALL, etc).

"""
import re
from os import walk
from pathlib import Path
import json

import pandas as pd

from subprocess import run
from fileseq import findSequencesOnDisk

__author__ = 'Joseph Goldstone'
__copyright__ = 'Copyright (C) 2020 Arnold & Richter Cine Technik GmbH & Co. Betriebs KG'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Joseph Goldstone'
__email__ = 'jgoldstone@arri.com'
__status__ = 'Experimental'

__all__ = [
    'Catalog'
]

MOV_EXTENSION = '.mov'
MXF_EXTENSION = '.mxf'
CLIP_EXTENSIONS = [MOV_EXTENSION, MXF_EXTENSION]
DPX_EXTENSION = '.dpx'
EXR_EXTENSION = '.exr'
ARI_EXTENSION = '.ari'
IMAGE_EXTENSIONS = [DPX_EXTENSION, EXR_EXTENSION, ARI_EXTENSION]
ROOT_DIRS_TO_SKIP = ['Applications', 'Clean', 'Clouds', 'Desktop', 'Documents', 'Downloads', 'Installers', 'Library',
                     'old_installers', 'Photos', 'Public', 'PyCharmProjects', 'VisualStudioCode', 'Xcode',
                     'repos', 'Music', 'bin', 'clones', 'lib', 'repos', 'vms']

EXTENSION_COL = 'extension'
EI_COL = 'exposure_index'
COLORSPACE_COL = 'colorspace'
DIR_PATH_COL = 'root_dir'
NAME_COL = 'name'
SIZE_COL = 'size'
SEQ_FRAME_WIDTH_COL = 'seq_frame_width'
START_COL = 'start_frame'
END_COL = 'end_frame'
INC_COL = 'frame_inc'
FIRST_FRAME_PATH_COL = 'first_frame_path'


# MAXFALL_COL = 'max_FALL'
# THUMB_COL = 'thumbnail'

class Catalog:

    def __init__(self, path):
        self._db_path = path
        self._db = None
        self._extension_census = {}
        self._subdir_roots = set()

    def load(self):
        """

        Parameters
        ----------

        Returns
        -------
        DataFrame
        """
        if self._db_path.exists():
            with open(self._db_path) as input_file:
                self._db = json.load(input_file)
        else:
            raise FileNotFoundError(f"expected frame catalog at `{self._db_path}' does not exist")

    def save_as(self, path: Path):
        with open(path, 'w') as output_file:
            self._db.to_csv(output_file)

    def save(self):
        assert self._db_path
        self.save_as(self._db_path)

    def save_by_extension_as(self, extension, db_path, verbose=False):
        """Save database on all clips or sequences with a given extension at a given path

        Parameters
        ----------
        extension : str
            An extension for a clip (e.g. .mxf, .mov) or an image sequence (e.g. .dpx, .exr)
        db_path : path_like
            Path of file to which database of clips or sequences of this extension should be saved
        verbose : bool
            If True, print statement of number of items saved.

        Returns
        -------

        """
        items = self._db[self._db[EXTENSION_COL] == extension]
        if not items.empty:
            with open(db_path, 'w') as output_file:
                items.to_csv(output_file)
                if verbose:
                    print(f"{extension}: saved {len(items):4} items")

    # def _census_by_extension(self, extension):
    #     if extension in self._extension_census:
    #         self._extension_census[extension] = self._extension_census[extension] + 1
    #     else:
    #         self._extension_census[extension] = 1

    # def _print_census_by_extension(self):
    #     for (extension, count) in self._extension_census.items():
    #         print(f"{extension}: {self._extension_census[extension]}")

    # def _register_subdir_root(self, dir_path):
    #     """Register a subdir iff we haven't registered a subdir higher up in the hierarchy
    #
    #     Parameters
    #     ----------
    #     dir_path
    #
    #     Returns
    #     -------
    #
    #     Notes
    #     -----
    #     Depends on representation of directories ending with directory separator
    #     """
    #     if not self._subdir_roots:
    #         self._subdir_roots.add(dir_path)
    #         return
    #     if dir_path in self._subdir_roots:
    #         return
    #     for subdir_root in self._subdir_roots:
    #         # it's deeper than an existing subdir root, ignore it
    #         if dir_path.startswith(subdir_root):
    #             # print(f"ignoring {root_dir}")
    #             return
    #     potentially_removing = True
    #     while potentially_removing:
    #         for subdir_root in self._subdir_roots:
    #             # It's higher than an existing subdir root, replace existing with this one
    #             if subdir_root.startswith(dir_path):
    #                 # print(f"removing {subdir_root} as it's deeper than {root_dir}")
    #                 self._subdir_roots.remove(subdir_root)
    #                 break
    #         potentially_removing = False
    #     # it's not here already
    #     # it's not deeper than something that's already here
    #     # if there was something below it, that something has been removed
    #     # so it's time to add it
    #     # print(f"adding {root_dir} to subdir root set")
    #     self._subdir_roots.add(dir_path)

    # def _print_subdir_roots(self):
    #     if self._subdir_roots:
    #         print("subdir roots:")
    #         for subdir_root in sorted(self._subdir_roots):
    #             print(f"\t{subdir_root}")

    def _get_actual_path_and_size(self, fileseq):
        if fileseq.extension() in CLIP_EXTENSIONS:
            clip_path = fileseq.frame(fileseq.start())
            return clip_path, Path(clip_path).stat().st_size
        if fileseq.extension() in IMAGE_EXTENSIONS:
            if fileseq.zfill() == 0:
                unique_path = fileseq.frame(fileseq.start())
                return unique_path, Path(unique_path).stat().st_size
            else:
                frame_set = fileseq.frameSet()
                paths = [fileseq[idx] for idx, fr in enumerate(fileseq.frameSet())]
                seq_size = 0
                for path in paths:
                    p = Path(path)
                    if p.exists():
                        seq_size += p.stat().st_size
                return fileseq.frame(frame_set.start()), seq_size

    def _get_colorspace_and_ei(self, path):
        """Return colorspace and exposure index for frame or clip at the supplied path

        Parameters
        ----------
        path : path_like
            path of clip or frame

        Returns
        -------
        colorspace : str
        exposure_index : int
        """
        error_pattern = None
        line_with_info = None
        pattern_for_info_line = None
        extension = path.extension()
        if extension == MXF_EXTENSION:
            pattern_for_ame_penultimate_line = r"\s*\d+\s+(\d+)\s+(\S+)\s*"
            groups_for_colorspace_and_ei = (2, 1)
            # don't bother trying to catch errors — ARRIMetaExtract_CMD always returns 0
            completed_process = run(
                ['ARRIMetaExtract_CMD', '-i', path, '-q', 'Target Color Space, Exposure Index ASA',
                 '-r', 'first'], capture_output=True)
        elif extension == MOV_EXTENSION:
            path = path.parent()
            # not that we ask for it, but ARRIMetaWExtract_CMD is going to tell us anyway
            pattern_for_ame_penultimate_line = r"\s*\d+\s+Apple ProRes (422 LT|422|422HQ|4444|4444 XQ)\s+(\d+)\s+(\S+)\s*"
            groups_for_colorspace_and_ei = (3, 2)
        elif extension == EXR_EXTENSION:
            path_for_ame = path
            pattern_for_ame_penultimate_line = ""
        elif extension == DPX_EXTENSION:
            pass
        elif extension == ARI_EXTENSION:
            pass
        return "foo", 5


        if completed_process.returncode == 0:
            lines = completed_process.stdout.decode().split('\n')
            penultimate_line = lines[-2]
            pattern = r"\s*\d+\s+(\d+)\s+(\S+)\s*"
            regex = re.compile(pattern)
            match = regex.match(penultimate_line)
            ei = match.group(1)
            colorspace = match.group(2)
            return colorspace, int(ei)
        else:
            print("failed!")

    def add_entry(self, fileseq):
        """

        Parameters
        ----------
        seq - a FileSequence object
        """
        extension = fileseq.extension()
        dir_path = fileseq.dirname()
        name = fileseq.basename().rstrip('.')
        start = fileseq.start()
        end = fileseq.end()
        inc = 1
        seq_frame_width = len(fileseq.padding())
        # colorspace, ei = self._get_colorspace_and_ei(Path(first_frame_path))
        first_frame_path, size = self._get_actual_path_and_size(fileseq)
        size = size // (1000 * 1000)
        if self._db is None:
            self._db = pd.DataFrame(columns=[EXTENSION_COL, EI_COL, COLORSPACE_COL, DIR_PATH_COL,
                                             NAME_COL, SEQ_FRAME_WIDTH_COL, START_COL,
                                             END_COL, INC_COL, SIZE_COL, FIRST_FRAME_PATH_COL])
        entry = {EXTENSION_COL: extension, EI_COL: pd.NA, COLORSPACE_COL: pd.NA, DIR_PATH_COL: dir_path,
                 NAME_COL: name, SEQ_FRAME_WIDTH_COL: seq_frame_width,
                 START_COL: start, END_COL: end, INC_COL: inc, SIZE_COL: size,
                 FIRST_FRAME_PATH_COL: first_frame_path}
        print(f"size: {size:7} MB, path {first_frame_path}")
        self._db = self._db.append(entry, ignore_index=True)

    def _register_sequences_in_dir(self, dir_path):
        """Register all sequences in a directory with extensions in either CLIP_EXTENSIONS or IMAGE_EXTENSIONS

        Parameters
        ----------
        dir_path : path_like
            directory in which clips or images are sought
        """
        seqs = findSequencesOnDisk(str(dir_path))
        for seq in seqs:
            extension = seq.extension()
            if extension in CLIP_EXTENSIONS or extension in IMAGE_EXTENSIONS:
                self.add_entry(seq)

    def _preened_subdirs(self, root_dir, subdirs):
        preened = []
        for subdir in subdirs:
            if subdir != root_dir and subdir not in ROOT_DIRS_TO_SKIP:
                preened.append(subdir)
        # TODO replace with following list expression
        # return [s for s in subdirs if s != root_dir and s not in ROOT_DIRS_TO_SKIP]
        return preened

    def register_content(self, root_dir):
        """Register all content with certain clip or frame suffixes under the given root directory

        Parameters
        ----------
        root_dir : path_like
            Path in which and underneath which clips or frames will be sought, subject to the subdirectories
            matching certain patterns, where clips or frames have certain extensions.
        """
        if not root_dir.exists():
            raise FileNotFoundError(f"no directory at {str(root_dir)}")
        if not root_dir.is_dir():
            raise FileNotFoundError(f"no directory at {str(root_dir)}, though some other type of file is there)")
        for this_dir, subdirs, _ in walk(root_dir):
            subdirs = self._preened_subdirs(root_dir, subdirs)
            for subdir in subdirs:
                self._register_sequences_in_dir(Path(this_dir) / subdir)
        for extension in CLIP_EXTENSIONS + IMAGE_EXTENSIONS:
            print(f"extension is {extension}")
            self.save_by_extension_as(extension, Path(root_dir)/f"registered_{extension.lstrip('.')}_items.csv",
                                      verbose=True)
