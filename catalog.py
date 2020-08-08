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

from os import walk
from pathlib import Path
import json

import pandas as pd

from fileseq import findSequencesOnDisk

__author__ = 'Joseph Goldstone'
__copyright__ = 'Copyright (C) 2020 Arnold & Richter Cine Technik GmbH & Co. Betriebs KG'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Joseph Goldstone'
__email__ = 'jgoldstone@arri.com'
__status__ = 'Experimental'

# TODO find out if class methods would show up in __all__
__all__ = [
    'Catalog'
]

USABLE_IMAGE_SUFFIXES = ['.ari', '.exr', '.mxf', '.mov', '.dpx']
# TODO really should check for context for these ($HOME for first two, $HOME/Library for last two)
ROOT_DIRS_TO_SKIP = ['Applications', 'Clouds', 'Desktop', 'Documents', 'Downloads', 'Library',
                     'Photos', 'Public', 'PyCharmProjects', 'VisualStudioCode', 'Xcode',
                     'repos', 'Music', 'bin', 'clones', 'lib', 'repos', 'vms']

SUFFIX_COL = 'suffix'
# COLORSPACE_COL = 'colorspace'
DIR_PATH_COL = 'dir_path'
NAME_COL = 'name'
# SIZE_COL = 'size'
SEQ_FRAME_WIDTH_COL = 'seq_frame_width'
START_COL = 'start_frame'
END_COL = 'end_frame'
INC_COL = 'frame_inc'


# MAXFALL_COL = 'max_FALL'
# THUMB_COL = 'thumbnail'

class Catalog:

    def __init__(self, path):
        self._db_path = path
        self._db = None
        self._suffix_census = {}
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

    def _census_by_suffix(self, suffix):
        if suffix in self._suffix_census:
            self._suffix_census[suffix] = self._suffix_census[suffix] + 1
        else:
            self._suffix_census[suffix] = 1

    def _print_census_by_suffix(self):
        for (suffix, count) in self._suffix_census.items():
            print(f"{suffix}: {self._suffix_census[suffix]}")

    def _register_subdir_root(self, dir_path):
        """Register a subdir iff we haven't registered a subdir higher up in the hierarchy

        Parameters
        ----------
        dir_path

        Returns
        -------

        Notes
        -----
        Depends on representation of directories ending with directory separator
        """
        if not self._subdir_roots:
            self._subdir_roots.add(dir_path)
            return
        if dir_path in self._subdir_roots:
            return
        for subdir_root in self._subdir_roots:
            # it's deeper than an existing subdir root, ignore it
            if dir_path.startswith(subdir_root):
                # print(f"ignoring {dir_path}")
                return
        potentially_removing = True
        while potentially_removing:
            for subdir_root in self._subdir_roots:
                # It's higher than an existing subdir root, replace existing with this one
                if subdir_root.startswith(dir_path):
                    # print(f"removing {subdir_root} as it's deeper than {dir_path}")
                    self._subdir_roots.remove(subdir_root)
                    break
            potentially_removing = False
        # it's not here already
        # it's not deeper than something that's already here
        # if there was something below it, that something has been removed
        # so it's time to add it
        # print(f"adding {dir_path} to subdir root set")
        self._subdir_roots.add(dir_path)

    def _print_subdir_roots(self):
        if self._subdir_roots:
            print("subdir roots:")
            for subdir_root in sorted(self._subdir_roots):
                print(f"\t{subdir_root}")

    def add_image_entry(self, fileseq):
        """

        Parameters
        ----------
        seq - a FileSequence object
        """
        suffix = fileseq.extension().lstrip('.')
        dir_path = fileseq.dirname().rstrip('/')
        name = fileseq.basename()
        start = fileseq.start()
        end = fileseq.end()
        inc = 1
        seq_frame_width = len(fileseq.padding())
        if self._db is None:
            self._db = pd.DataFrame(columns=[SUFFIX_COL, DIR_PATH_COL, NAME_COL, SEQ_FRAME_WIDTH_COL, START_COL, END_COL, INC_COL])
        print('first entry!')
        entry = {SUFFIX_COL: suffix, DIR_PATH_COL: dir_path, NAME_COL: name, SEQ_FRAME_WIDTH_COL: seq_frame_width,
                 START_COL: start, END_COL: end, INC_COL: inc}
        self._db = self._db.append(entry, ignore_index=True)
        print(f"len(self._db) is {len(self._db)}")

    def register_content(self, dir_path):
        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"no directory at {str(path)}")
        if not path.is_dir():
            raise FileNotFoundError(f"no directory at {str(path)}, though some other type of file is there)")
        for walk_root, walk_dirnames, _ in walk(dir_path):
            if walk_root == dir_path:
                for root_dir_to_skip in ROOT_DIRS_TO_SKIP:
                    if root_dir_to_skip in walk_dirnames:
                        walk_dirnames.remove(root_dir_to_skip)
            walk_dirnames_to_remove = []
            for walk_dirname in walk_dirnames:
                if walk_dirname.startswith('.'):
                    walk_dirnames_to_remove.append(walk_dirname)
            for walk_dirname_to_remove in walk_dirnames_to_remove:
                walk_dirnames.remove(walk_dirname_to_remove)
            seqs = findSequencesOnDisk(walk_root)
            for seq in seqs:
                # print(seq)
                suffix = seq.extension()
                if suffix in USABLE_IMAGE_SUFFIXES:
                    self._census_by_suffix(suffix)
                    seq_dirname = seq.dirname()
                    if seq_dirname.rstrip('/') != dir_path.rstrip('/'):
                        self._register_subdir_root(seq_dirname)
                    self.add_image_entry(seq)
        self._print_census_by_suffix()
        self._print_subdir_roots()
