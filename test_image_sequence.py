import unittest
import uuid
from pathlib import Path
from typing import Iterable
from image_sequence import ImageSequence, _contiguous_ranges
import numpy as np

TEST_DIR_PATH = f"/tmp/py_unit_{uuid.uuid1()}"
TEST_FILE_NAME = "foo"
TEST_FILE_SUFFIX = "exr"
TEST_SEQ_START = 0
TEST_SEQ_END = 3
TEST_SEQ_INC = 1
TEST_SEQ_FRAME_RANGE = range(TEST_SEQ_START, TEST_SEQ_END, TEST_SEQ_INC)
TEST_SEQ_FRAME_DIGITS = 3


class MyTestCase(unittest.TestCase):

    def setUp(self):
        # rather dubious to have the test suite setup depend on something it's defining.
        # TODO re-implement test_image_sequence.py setUp() and tearDown() without using ImageSequence itself
        self.test_seq = ImageSequence(TEST_DIR_PATH, TEST_FILE_NAME, TEST_FILE_SUFFIX, range(TEST_SEQ_START, TEST_SEQ_END, TEST_SEQ_INC), TEST_SEQ_FRAME_DIGITS)
        d = Path(self.test_seq.dir_path)
        if not d.exists():
            d.mkdir(mode=0o777)
            print(f"created directory `{d}'")
        else:
            raise RuntimeError(f"cannot create directory `{d}' for some reason")
        for f in range(TEST_SEQ_START, TEST_SEQ_END, TEST_SEQ_INC):
            p = self.test_seq.path_for_frame(f)
            print(f"\tcreated file `{p}'")
            p.touch()

    def recursive_remove_dir_contents(self, fs_dir, indentation_level):
        indentation = '\t' * indentation_level
        dir_contents = fs_dir.glob('*')
        for content in dir_contents:
            if content.is_dir():
                self.recursive_remove_dir_contents(content, indentation_level + 1)
                content.rmdir()
                print(f"{indentation}removed directory `{content}'")
            else:
                content.unlink()
                print(f"{indentation}removed file `{content}'")

    def tearDown(self):
        d = Path(self.test_seq.dir_path)
        self.recursive_remove_dir_contents(d, 1)
        d.rmdir()
        print(f"removed directory `{d}"'')

    def test_ctor(self):
        self.assertEqual(TEST_DIR_PATH, str(self.test_seq.dir_path))
        self.assertEqual(TEST_FILE_NAME, self.test_seq.name)
        self.assertEqual(TEST_FILE_SUFFIX, self.test_seq.suffix)
        self.assertEqual(TEST_SEQ_FRAME_RANGE, self.test_seq.frame_range)
        self.assertEqual(TEST_SEQ_FRAME_DIGITS, self.test_seq.frame_digits)

    def test_path_for_frame(self):
        seq = ImageSequence(TEST_DIR_PATH, TEST_FILE_NAME, TEST_FILE_SUFFIX, range(TEST_SEQ_START, TEST_SEQ_END, TEST_SEQ_INC), TEST_SEQ_FRAME_DIGITS)
        path_for_frame = seq.path_for_frame(1)
        self.assertTrue(f"{TEST_DIR_PATH}/{TEST_FILE_NAME}.001.exr" == str(path_for_frame))

    @staticmethod
    def make_subdir_with_sequence(dir_path: str, base_name: str, seq: Iterable, frame_number_width: int, suffix: str) -> Path:
        """Make a temporary subdirectory directory with 0-length frames according to supplied arguments

        Parameters
        ----------
        dir_path : Path
            Path in which subdirs will be created
        base_name : unicode
            Filename component of the frames to be generated, or None
        seq : sequence of int
            Frame numbers for the frames in the generated sequence;
            must be non-negative integers
        frame_number_width : int, or None
            Width to which the frame number field should be zero-padded; must be positive integer
        suffix : unicode or None
            Suffix component of frames to be generated

        Returns
        -------
        subdir_path
            Path in which the desired frames have been created

        Raises
        ------
        AssertionError
            If any supplied frame number is negative, or the specified frame number width is negative,
            or any frame numbers are supplied but the frame humber width is None.

        Notes
        -----
        If the base name is None
        If the frame number width is None, no extra period is inserted prior
            to where the frame number would normally appear.
        If the suffix is None, no extra period is inserted prior to where the
            suffix would normally appear.
        """
        dir_path_as_path = Path(dir_path)
        tmp_subdir_path = dir_path_as_path / str(uuid.uuid1())
        assert tmp_subdir_path is not dir_path_as_path
        if frame_number_width is not None:
            assert frame_number_width > 0
        else:
            assert not seq
        tmp_subdir_path.mkdir()
        if seq:
            assert sum(np.array(seq) < 0) == 0
            for frame_number in seq:
                name = ""
                if base_name is not None:
                    name = base_name + "."
                assert len(str(frame_number)) <= frame_number_width
                name = name + f"{frame_number:0{frame_number_width}}"
                if suffix is not None:
                    name = name + f".{suffix}"
                tmp_file_path = tmp_subdir_path / name
                tmp_file_path.touch()
        return tmp_subdir_path

    # def remove_dir_and_any_contained_files(self, root_dir):
    #     assert(root_dir.exists())
    #     assert(root_dir.is_dir())
    #     contained_files = dir_path_exists.glob('*')
    #     for contained_file in contained_files:
    #         contained_file.unlink()
    #     root_dir.rmdir()

    # unit tests for unit test helper function!
    def test_msws_raises_on_0_width(self):
        self.assertRaises(AssertionError, self.make_subdir_with_sequence, TEST_DIR_PATH, "x", [0, 1, 2, 3], 0, "exr")

    def test_msws_raises_on_neg_frame_numbers(self):
        self.assertRaises(AssertionError, self.make_subdir_with_sequence, TEST_DIR_PATH, "x", [0, 1, 2, -3], 1, "exr")

    def test_frame_field_overflow_throws(self):
        self.assertRaises(AssertionError, self.make_subdir_with_sequence, TEST_DIR_PATH, "x", [0, 11, 2, 3], 1, "exr")

    def test_frame_neg_frame_number_width_throws(self):
        self.assertRaises(AssertionError, self.make_subdir_with_sequence, TEST_DIR_PATH, "x", [0, 11, 2, 3], -1, "exr")

    def test_frame_seq_no_frame_numbers_valid_frame_width(self):
        tmp_subdir_path = self.make_subdir_with_sequence(TEST_DIR_PATH, "x", [], 4, "exr")
        files = tmp_subdir_path.glob('*')
        self.assertEqual(0, len(list(files)))

    def test_frame_seq_odd_numbers_below_8(self):
        tmp_subdir_path = self.make_subdir_with_sequence(TEST_DIR_PATH, "x", [1, 3, 5, 7], 4, "exr")
        files = tmp_subdir_path.glob('*.exr')
        self.assertEqual(4, len(list(files)))
        self.assertTrue(Path(tmp_subdir_path / 'x.0001.exr').exists())
        self.assertTrue(Path(tmp_subdir_path / 'x.0003.exr').exists())
        self.assertTrue(Path(tmp_subdir_path / 'x.0005.exr').exists())
        self.assertTrue(Path(tmp_subdir_path / 'x.0007.exr').exists())

    def test_contiguous_range_single_range(self):
        reference = range(2,7)
        actual = _contiguous_ranges(list(reference))[0]
        self.assertEqual(reference, actual)

    def _check_round_trip(self, dir_path, name, suffix, frame_ranges, frame_digits):
        flattened_frame_seq = []
        for frame_range in frame_ranges:
            flattened_frame_seq.extend(frame_range)
        candidates = ImageSequence.image_sequences_in_dir(dir_path)
        candidate_flattened_frame_seq = []
        for candidate in candidates:
            self.assertEqual(dir_path, candidate.dir_path)
            self.assertEqual(name, candidate.name)
            self.assertEqual(suffix, candidate.suffix)
            candidate_flattened_frame_seq.extend(candidate.frame_range)
            self.assertEqual(frame_digits, candidate.frame_digits)
        self.assertEqual(flattened_frame_seq, candidate_flattened_frame_seq)

    # foo.0001.exr should produce [('foo', Range(1,2), 4, "exr")]
    def test_single_singleton_sequence(self):
        self._check_round_trip(TEST_DIR_PATH, TEST_FILE_NAME, TEST_FILE_SUFFIX, [range(1, 2)], 4)

    def test_single_adjacent_sequence(self):
        self._check_round_trip(TEST_DIR_PATH, TEST_FILE_NAME, TEST_FILE_SUFFIX, [range(1, 7)], 4)

    def test_adjacent_singleton_and_continuous_sequence(self):
        self._check_round_trip(TEST_DIR_PATH, TEST_FILE_NAME, TEST_FILE_SUFFIX, [range(1, 2), range(3, 7)], 4)

    def test_adjacent_sequences(self):
        self._check_round_trip(TEST_DIR_PATH, TEST_FILE_NAME, TEST_FILE_SUFFIX, [range(1, 3), range(5, 7)], 4)

    def test_adjacent_continuous_sequence_and_singleton(self):
        self._check_round_trip(TEST_DIR_PATH, TEST_FILE_NAME, TEST_FILE_SUFFIX, [range(1, 5), range(6, 7)], 4)

if __name__ == '__main__':
    unittest.main()
