import unittest
from pathlib import Path
from image_sequence import ImageSequence

TEST_DIR_PATH = "/tmp"
TEST_FILE_NAME = "foo"
TEST_FILE_EXT = "exr"
TEST_SEQ_START = 0
TEST_SEQ_END = 2
TEST_SEQ_INC = 1
TEST_SEQ_FRAME_DIGITS = 3


class MyTestCase(unittest.TestCase):
    def test_ctor(self):
        seq = ImageSequence(TEST_DIR_PATH, TEST_FILE_NAME, TEST_FILE_EXT, start=TEST_SEQ_START, end=TEST_SEQ_END,
                            inc=TEST_SEQ_INC, frame_digits=TEST_SEQ_FRAME_DIGITS)
        self.assertEqual(TEST_DIR_PATH, seq._dir_path)
        self.assertEqual(TEST_FILE_NAME, seq._file_name)
        self.assertEqual(TEST_FILE_EXT, seq._file_ext)
        self.assertEqual(TEST_SEQ_START, seq._start)
        self.assertEqual(TEST_SEQ_END, seq._end)
        self.assertEqual(TEST_SEQ_INC, seq._inc)
        self.assertEqual(TEST_SEQ_FRAME_DIGITS, seq._frame_digits)

    def test_path_for_frame(self):
        seq = ImageSequence(TEST_DIR_PATH, TEST_FILE_NAME, TEST_FILE_EXT, start=TEST_SEQ_START, end=TEST_SEQ_END,
                            inc=TEST_SEQ_INC, frame_digits=TEST_SEQ_FRAME_DIGITS)
        path_for_frame = seq.path_for_frame(1)
        self.assertTrue (Path("/tmp/foo.001.exr") == path_for_frame)

if __name__ == '__main__':
    unittest.main()
