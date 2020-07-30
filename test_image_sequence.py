import unittest
import uuid
from pathlib import Path
from image_sequence import ImageSequence
import numpy as np

TEST_DIR_PATH = f"/tmp/py_unit_{uuid.uuid1()}"
TEST_FILE_NAME = "foo"
TEST_FILE_SUFFIX = "exr"
TEST_SEQ_START = 0
TEST_SEQ_END = 3
TEST_SEQ_INC = 1
TEST_SEQ_FRAME_DIGITS = 3


class MyTestCase(unittest.TestCase):

    def setUp(self):
        # rather dubious to have the test suite setup depend on something it's defining.
        # TODO re-implement test_image_sequence.py setUp() and tearDown() without using ImageSequence itself
        self.test_seq = ImageSequence(TEST_DIR_PATH, TEST_FILE_NAME, TEST_FILE_SUFFIX, start=TEST_SEQ_START, end=TEST_SEQ_END, inc=TEST_SEQ_INC, frame_digits=TEST_SEQ_FRAME_DIGITS)
        d = Path(self.test_seq._dir)
        if not d.exists():
            d.mkdir(mode=0o777)
            print(f"created directory `{d}'")
        else:
            raise RuntimeError(f"cannot create directory `{d}' for some reason")
        for f in range(TEST_SEQ_START, TEST_SEQ_END, TEST_SEQ_INC):
            p = self.test_seq.path_for_frame(f)
            print(f"\tcreated file `{p}'")
            p.touch()

    def recursive_remove_dir_contents(self, dir, indentation_level):
        indentation = '\t'*indentation_level
        dir_contents = dir.glob('*')
        for content in dir_contents:
            if content.is_dir():
                self.recursive_remove_dir_contents(content, indentation_level + 1)
                content.rmdir()
                print(f"{indentation}removed directory `{content}'")
            else:
                content.unlink()
                print(f"{indentation}removed file `{content}'")


    def tearDown(self):
        d = Path(self.test_seq._dir)
        self.recursive_remove_dir_contents(d, 1)
        d.rmdir()
        print(f"removed directory `{d}"'')

    def test_ctor(self):
        self.assertEqual(TEST_DIR_PATH, str(self.test_seq.dir))
        self.assertEqual(TEST_FILE_NAME, self.test_seq._name)
        self.assertEqual(TEST_FILE_SUFFIX, self.test_seq._suffix)
        self.assertEqual(TEST_SEQ_START, self.test_seq.start)
        self.assertEqual(TEST_SEQ_END, self.test_seq.end)
        self.assertEqual(TEST_SEQ_INC, self.test_seq.inc)
        self.assertEqual(TEST_SEQ_FRAME_DIGITS, self.test_seq._frame_digits)

    def test_path_for_frame(self):
        seq = ImageSequence(TEST_DIR_PATH, TEST_FILE_NAME, TEST_FILE_SUFFIX, start=TEST_SEQ_START, end=TEST_SEQ_END,
                            inc=TEST_SEQ_INC, frame_digits=TEST_SEQ_FRAME_DIGITS)
        path_for_frame = seq.path_for_frame(1)
        self.assertTrue(f"{TEST_DIR_PATH}/{TEST_FILE_NAME}.001.exr" == str(path_for_frame))


    def make_subdir_with_sequence(self, dir_path, base_name, seq, frame_number_width, suffix):
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
        if base_name is not None:
            name = base_name
        if frame_number_width is not None:
            assert frame_number_width > 0
        else:
            assert len(seq) == 0
        tmp_subdir_path.mkdir()
        if len(seq) != 0:
            assert sum(np.array(seq) < 0) == 0
            for frame_number in seq:
                if base_name is not None:
                    name = name + "."
                assert len(str(frame_number)) <= frame_number_width
                name = name + f"{frame_number:0{frame_number_width}}"
                if suffix is not None:
                    name = name + f".{suffix}"
                tmp_file_path = tmp_subdir_path / name
                tmp_file_path.touch()
        return tmp_subdir_path

    # def remove_dir_and_any_contained_files(self, dir_path):
    #     assert(dir_path.exists())
    #     assert(dir_path.is_dir())
    #     contained_files = dir_path_exists.glob('*')
    #     for contained_file in contained_files:
    #         contained_file.unlink()
    #     dir_path.rmdir()

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



if __name__ == '__main__':
    unittest.main()
