import unittest
import uuid
from pathlib import Path
from image_sequence import ImageSequence

TEST_DIR_PATH = f"/tmp/py_unit_{uuid.uuid1()}"
TEST_FILE_NAME = "foo"
TEST_FILE_EXT = "exr"
TEST_SEQ_START = 0
TEST_SEQ_END = 3
TEST_SEQ_INC = 1
TEST_SEQ_FRAME_DIGITS = 3


class MyTestCase(unittest.TestCase):

    def setUp(self):
        # rather dubious to have the test suite setup depend on something it's defining.
        # TODO re-implement test_image_sequence.py setUp() and tearDown() without using ImageSequence itself
        self.test_seq = ImageSequence(TEST_DIR_PATH, TEST_FILE_NAME, TEST_FILE_EXT, start=TEST_SEQ_START, end=TEST_SEQ_END, inc=TEST_SEQ_INC, frame_digits=TEST_SEQ_FRAME_DIGITS)
        d = Path(self.test_seq._dir)
        if not d.exists():
            d.mkdir(mode=0o777)
            print(f"created directory `{d}'")
        else:
            raise RuntimeError(f"cannot create dir at `{d}' for some reason")
        for f in range(TEST_SEQ_START, TEST_SEQ_END, TEST_SEQ_INC):
            p = self.test_seq.path_for_frame(f)
            print(f"\tcreated file `{p}'")
            p.touch()

    def tearDown(self):
        for f in range(TEST_SEQ_START, TEST_SEQ_END, TEST_SEQ_INC):
            p = self.test_seq.path_for_frame(f)
            p.unlink(missing_ok=True)
            print(f"\tremoved file `{p}'")
        d = Path(self.test_seq._dir)
        d.rmdir()
        print(f"removed directory `{d}'")

    def test_ctor(self):
        self.assertEqual(TEST_DIR_PATH, str(self.test_seq.dir))
        self.assertEqual(TEST_FILE_NAME, self.test_seq._name)
        self.assertEqual(TEST_FILE_EXT, self.test_seq._suffix)
        self.assertEqual(TEST_SEQ_START, self.test_seq.start)
        self.assertEqual(TEST_SEQ_END, self.test_seq.end)
        self.assertEqual(TEST_SEQ_INC, self.test_seq.inc)
        self.assertEqual(TEST_SEQ_FRAME_DIGITS, self.test_seq._frame_digits)

    def test_path_for_frame(self):
        seq = ImageSequence(TEST_DIR_PATH, TEST_FILE_NAME, TEST_FILE_EXT, start=TEST_SEQ_START, end=TEST_SEQ_END,
                            inc=TEST_SEQ_INC, frame_digits=TEST_SEQ_FRAME_DIGITS)
        path_for_frame = seq.path_for_frame(1)
        self.assertTrue(f"{TEST_DIR_PATH}/{TEST_FILE_NAME}.001.exr" == str(path_for_frame))

    def test_frame_field_overflow_throws(self):
        pass

if __name__ == '__main__':
    unittest.main()
