import unittest
from characterization_sequence import CharacterizationSequence


class C18nSequenceTestCase(unittest.TestCase):

    def test_sequence_loads(self):
        frame_paths_to_test = ['/tmp/foo.01.exr', '/tmp/foo.00.exr', '/tmp/foo.02.exr']
        for frame_path in frame_paths_to_test:
            open(frame_path, 'a').close()
        c18n = CharacterizationSequence('/tmp', 'foo', 2, 0, 2)
        expected = sorted(frame_paths_to_test)
        actual = [str(fp) for fp in sorted(c18n.frame_paths)]
        self.assertEqual(expected, actual)

    def xtest_real_sequence_subset_loads(self):
        dir_path = '/Users/jgoldstone/Content/not_secret_content/arri/bur/tfe/aces/vwg_gm/balloon+2XL5' \
                   '-C_demo_mode_test/derived/seqs/A003R3VC/A003C003_120101_cropped'
        file_base = 'A003C003_120101_cropped'
        frame_number_width = 4
        first = 1
        last = 5
        c18n = CharacterizationSequence(dir_path, file_base, frame_number_width, first, last)
        self.assertEqual(5, len(c18n.frame_paths))
        c18n.characterize_frames()


if __name__ == '__main__':
    unittest.main()
