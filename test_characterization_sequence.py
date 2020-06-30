import unittest

from characterization_sequence import CharacterizationSequence


class C18nSequenceTestCase(unittest.TestCase):

    @staticmethod
    def test_sequence_loads():
        for frame_path in ['/tmp/foo.00.exr', '/tmp/foo.01.exr', '/tmp/foo.02.exr']:
            open(frame_path, 'a').close()
        CharacterizationSequence('/tmp', 'foo', 2, 0, 2)


if __name__ == '__main__':
    unittest.main()
