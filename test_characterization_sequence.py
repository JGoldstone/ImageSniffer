import unittest

from image_sequence import ImageSequence

from characterization_sequence import CharacterizationSequence

HOT_DIR = "/Users/jgoldstone/Content/not_secret_content/arri/bur/tfe/aces/vwg_gm/balloon+2XL5-C_demo_mode_test/derived/seqs/A003R3VC/A003C003_120101_cropped"
HOT_FILE_NAME = "A003C003_120101_cropped"
HOT_EXT = "exr"
# start at 1, end at 7212
# HOT_FIRST = 1000
# HOT_LAST = 1050
HOT_FIRST = 50
HOT_LAST = 60
HOT_INC = 1
HOT_FRAME_DIGITS = 4

class C18nSequenceTestCase(unittest.TestCase):

    def setUp(self):
        self.hot_seq = ImageSequence(HOT_DIR, HOT_FILE_NAME, HOT_EXT, HOT_FIRST, HOT_LAST, HOT_INC, HOT_FRAME_DIGITS)

    def test_sequence_exists(self):
        for f in range(HOT_FIRST, HOT_LAST, HOT_INC):
            p = self.hot_seq.path_for_frame(f)
            open(p).close()

    def test_seq_c18ns_load(self):
        c18n_seq = CharacterizationSequence(HOT_DIR, HOT_FILE_NAME, HOT_FRAME_DIGITS, HOT_FIRST, HOT_LAST)
        c18n_seq.build_c18n_list()
        self.assertEqual(len(c18n_seq.frame_paths), len(c18n_seq.c18ns))

if __name__ == '__main__':
    unittest.main()
